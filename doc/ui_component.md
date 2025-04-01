##

<table width="100%" >
  <tr>
    <td align="center" width=30%><strong>Component for basic</strong></td>
    <td align="center"><strong>describe</strong></td>
  <tr>
  <tr>
    <td align="center" width=30%>Lang</td>
    <td align="center">Choose language</td>
  <tr>
  <tr>
    <td align="center" width=30%>Model name</td>
    <td align="center">Choose ViT-B/16 or ViT-B/32 for clip<br>Choose Custom for adaclip</td>
  <tr>
  <tr>
    <td align="center" width=30%>Model path</td>
    <td align="center">Fill in the absolute path of the file for adaclip</td>
  <tr>
  <tr>
    <td align="center" width=30%>Finetuning method</td>
    <td align="center">Choose clip or adaclip</td>
  <tr>
  <tr>
    <td align="center" width=30%>Stage</td>
    <td align="center">Choose clip or adaclip</td>
  <tr>
  <tr>
    <td align="center" width=30%>Data dir</td>
    <td align="center">Fill in the absolute path of the dataset directory</td>
  <tr>
  <tr>
    <td align="center" width=30%>use xpu</td>
    <td align="center">Select when train on Arc A770</td>
  <tr>
  <tr>
    <td align="center" width=30%>Batch size</td>
    <td align="center">Batch size</td>
  <tr>

</table>

##

<table width="100%" >
  <tr>
    <td align="center" width=30%><strong>Component for Clip_Finetune_tab</strong></td>
    <td align="center"><strong>describe</strong></td>
  <tr>
  <tr>
    <td align="center" width=30%>Traing group</td>
    <td align="center">Finetune or Training free</td>
  <tr>
  <tr>
    <td align="center" width=30%>clip_finetune method</td>
    <td align="center">related method</td>
  <tr>
  <tr>
    <td align="center" width=30%>
clip_finetune few_shot_num</td>
    <td align="center">used for few shot learning</td>
  <tr>
  <tr>
    <td align="center" width=30%>dataset_config_file</td>
    <td align="center">where to find dataset config file(no need to change)</td>
  <tr>
  <tr>
    <td align="center" width=30%>trainer config_file directory</td>
    <td align="center">where to find trainer config file(no need to change)</td>
  <tr>
</table>

##

<table width="100%" >
  <tr>
    <td align="center" width=30%><strong>Component for Adaclip_finetune_tab</strong></td>
    <td align="center"><strong>describe</strong></td>
  <tr>
  <tr>
    <td align="center" width=30%>Adaclip top_k</td>
    <td align="center">top k number</td>
  <tr>
  <tr>
    <td align="center" width=30%>Adaclip config file path</td>
    <td align="center">where to find trainer config file(no need to change)</td>
  <tr>
  <tr>
    <td align="center" width=30%>Adaclip freeze_cnn</td>
    <td align="center">whether to freeze cnn</td>
  <tr>
  <tr>
    <td align="center" width=30%>Adaclip frame_agg</td>
    <td align="center">Adaclip frame_agg</td>
  <tr>
</table>

##

<table width="100%" >
  <tr>
    <td align="center" width=30%><strong>Component for optuna configuration</strong></td>
    <td align="center"><strong>describe</strong></td>
  <tr>
  <tr>
    <td align="center" width=30%>use optuna</td>
    <td align="center">whether to use optuna</td>
  <tr>
  <tr>
    <td align="center" width=30%>number of trained times</td>
    <td align="center">number of trained times</td>
  <tr>
  <tr>
    <td align="center" width=30%>n_warmup_steps</td>
    <td align="center">number of warmup steps</td>
  <tr>
  <tr>
    <td align="center" width=30%>optuna sampler</td>
    <td align="center">sampler use in optuna</td>
  <tr>
  <tr>
    <td align="center" width=30%>self-defined optuna train param</td>
    <td align="center">self-defined optuna train param</td>
  <tr>
</table>
