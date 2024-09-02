import os
import tempfile
from functools import partial
from pathlib import Path
from typing import List, Optional, Union

from huggingface_hub import RepoUrl
from huggingface_hub.hf_api import CommitInfo, future_compatible
from requests.exceptions import HTTPError
from transformers.utils import logging, strtobool
from ..extras.misc import use_modelscope

logger = logging.get_logger(__name__)


def create_repo(repo_id: str, *, token: Union[str, bool, None] = None, private: bool = False, **kwargs) -> RepoUrl:
    from modelscope.hub.repository import Repository
    hub_model_id = PushToMsHubMixin.create_ms_repo(repo_id, token, private)
    PushToMsHubMixin.ms_token = token
    with tempfile.TemporaryDirectory() as temp_cache_dir:
        repo = Repository(temp_cache_dir, hub_model_id)
        PushToMsHubMixin.add_patterns_to_gitattributes(repo, ['*.safetensors', '*.bin', '*.pt'])
        # Add 'runs/' to .gitignore, ignore tensorboard files
        PushToMsHubMixin.add_patterns_to_gitignore(repo, ['runs/', 'images/'])
        PushToMsHubMixin.add_patterns_to_file(
            repo,
            'configuration.json', ['{"framework": "pytorch", "task": "text-generation", "allow_remote": true}'],
            ignore_push_error=True)
        # Add '*.sagemaker' to .gitignore if using SageMaker
        if os.environ.get('SM_TRAINING_ENV'):
            PushToMsHubMixin.add_patterns_to_gitignore(repo, ['*.sagemaker-uploading', '*.sagemaker-uploaded'],
                                                       'Add `*.sagemaker` patterns to .gitignore')
    return RepoUrl(url=hub_model_id, )


@future_compatible
def upload_folder(
    self,
    *,
    repo_id: str,
    folder_path: Union[str, Path],
    path_in_repo: Optional[str] = None,
    commit_message: Optional[str] = None,
    commit_description: Optional[str] = None,
    token: Union[str, bool, None] = None,
    revision: Optional[str] = 'master',
    ignore_patterns: Optional[Union[List[str], str]] = None,
    run_as_future: bool = False,
    **kwargs,
):
    from modelscope import push_to_hub
    commit_message = commit_message or 'Upload folder using api'
    if commit_description:
        commit_message = commit_message + '\n' + commit_description
    if not os.path.exists(os.path.join(folder_path, 'configuration.json')):
        with open(os.path.join(folder_path, 'configuration.json'), 'w') as f:
            f.write('{"framework": "pytorch", "task": "text-generation", "allow_remote": true}')
    if ignore_patterns:
        ignore_patterns = [p for p in ignore_patterns if p != '_*']
    if path_in_repo:
        # We don't support part submit for now
        path_in_repo = os.path.basename(folder_path)
        folder_path = os.path.dirname(folder_path)
        logger.warn(f'ModelScope does not support submitting a part of the sub-folders, all files in {folder_path} will be submitted.')
        ignore_patterns = []
    push_to_hub(
        repo_id,
        folder_path,
        token or PushToMsHubMixin.ms_token,
        commit_message=commit_message,
        ignore_file_pattern=ignore_patterns,
        revision=revision,
        tag=path_in_repo)
    return CommitInfo(
        commit_url=f'https://www.modelscope.cn/models/{repo_id}/files',
        commit_message=commit_message,
        commit_description=commit_description,
        oid=None,
    )


class PushToMsHubMixin:

    _use_hf_hub = not use_modelscope()
    ms_token = None

    if not _use_hf_hub:
        import huggingface_hub
        from huggingface_hub.hf_api import api
        from transformers import trainer
        huggingface_hub.create_repo = create_repo
        huggingface_hub.upload_folder = partial(upload_folder, api)
        trainer.create_repo = create_repo
        trainer.upload_folder = partial(upload_folder, api)

    @staticmethod
    def create_ms_repo(hub_model_id: str, hub_token: Optional[str] = None, hub_private_repo: bool = False) -> str:
        from modelscope import HubApi
        from modelscope.hub.api import ModelScopeConfig
        from modelscope.hub.constants import ModelVisibility
        assert hub_model_id is not None, 'Please enter a valid hub_model_id'

        api = HubApi()
        if hub_token is None:
            hub_token = os.environ.get('MODELSCOPE_API_TOKEN')
        if hub_token is not None:
            api.login(hub_token)
        else:
            raise ValueError('Please specify a token by `--hub_token` or `MODELSCOPE_API_TOKEN=xxx`')
        visibility = ModelVisibility.PRIVATE if hub_private_repo else ModelVisibility.PUBLIC

        if '/' not in hub_model_id:
            user_name = ModelScopeConfig.get_user_info()[0]
            assert isinstance(user_name, str)
            hub_model_id = f'{user_name}/{hub_model_id}'
            logger.info(f"'/' not in hub_model_id, pushing to personal repo {hub_model_id}")
        try:
            api.create_model(hub_model_id, visibility)
        except HTTPError:
            # The remote repository has been created
            pass
        return hub_model_id

    @staticmethod
    def add_patterns_to_file(repo,
                             file_name: str,
                             patterns: List[str],
                             commit_message: Optional[str] = None,
                             ignore_push_error=False) -> None:
        if isinstance(patterns, str):
            patterns = [patterns]
        if commit_message is None:
            commit_message = f'Add `{patterns[0]}` patterns to {file_name}'

        # Get current file content
        repo_dir = repo.model_dir
        file_path = os.path.join(repo_dir, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                current_content = f.read()
        else:
            current_content = ''
        # Add the patterns to file
        content = current_content
        for pattern in patterns:
            if pattern not in content:
                if len(content) > 0 and not content.endswith('\n'):
                    content += '\n'
                content += f'{pattern}\n'

        # Write the file if it has changed
        if content != current_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                logger.debug(f'Writing {file_name} file. Content: {content}')
                f.write(content)
        try:
            repo.push(commit_message)
        except Exception as e:
            if ignore_push_error:
                pass
            else:
                raise e

    @staticmethod
    def add_patterns_to_gitignore(repo, patterns: List[str], commit_message: Optional[str] = None) -> None:
        PushToMsHubMixin.add_patterns_to_file(repo, '.gitignore', patterns, commit_message, ignore_push_error=True)

    @staticmethod
    def add_patterns_to_gitattributes(repo, patterns: List[str], commit_message: Optional[str] = None) -> None:
        new_patterns = []
        suffix = 'filter=lfs diff=lfs merge=lfs -text'
        for pattern in patterns:
            if suffix not in pattern:
                pattern = f'{pattern} {suffix}'
            new_patterns.append(pattern)
        file_name = '.gitattributes'
        if commit_message is None:
            commit_message = f'Add `{patterns[0]}` patterns to {file_name}'
        PushToMsHubMixin.add_patterns_to_file(repo, file_name, new_patterns, commit_message, ignore_push_error=True)
