import os
import sys
import site

def get_python_executable():
    """
    Get the path of the current Python interpreter.
    """
    return sys.executable

def construct_modeling_qwen2_vl_path():
    """
    Construct the absolute path to transformers/models/qwen2_vl/modeling_qwen2_vl.py.
    """
    python_path = get_python_executable()
    print(f"Python executable path: {python_path}")
    
    # Get the site-packages directory
    site_packages = site.getsitepackages()
    if not site_packages:
        return "Cannot find site-packages directory"
    
    # Assume using the first site-packages path
    site_packages_path = site_packages[0]
    
    # Construct the path to the target file
    modeling_path = os.path.join(
        site_packages_path,
        'transformers',
        'models',
        'qwen2_vl',
        'modeling_qwen2_vl.py'
    )
    
    if os.path.exists(modeling_path):
        return os.path.abspath(modeling_path)
    else:
        return f"file does not exist: {modeling_path}"

def main():
    modeling_path = construct_modeling_qwen2_vl_path()
    print(f"Absolute path of modeling_qwen2_vl.py: {modeling_path}")

if __name__ == "__main__":
    main()