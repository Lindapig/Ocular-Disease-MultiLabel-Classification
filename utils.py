import os
import pandas as pd


def read_file(filepath):
    """
    Reads a CSV or Excel file based on its file extension.

    Args:
        filepath (str): The path to the file.

    Returns:
        pd.DataFrame: DataFrame containing the data.
    """
    # Get the file extension
    file_extension = os.path.splitext(filepath)[-1].lower()

    # Read the file based on the extension
    if file_extension == ".csv":
        return pd.read_csv(filepath)
    elif file_extension in [".xls", ".xlsx"]:
        return pd.read_excel(filepath, sheet_name="Sheet1")
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")


def list_image_files(folder_path):
    # Supported image file extensions
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp")

    # List all files in the folder with the supported extensions
    image_files = [
        file
        for file in os.listdir(folder_path)
        if file.lower().endswith(image_extensions)
    ]

    return image_files
