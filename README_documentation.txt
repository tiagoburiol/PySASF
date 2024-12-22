PySASF Documentation Guide

    Welcome to the documentation script for the PySASF package. This package utilizes Sphinx's quickstart feature to generate documentation directly from the docstrings embedded within the code. Below, you will find a comprehensive step-by-step guide tailored to your specific use case for reading, generating, or updating the documentation.


Important Note for All Use Cases:
    Please refrain from placing any important files within the "PySASF/docs" folder! They may be deleted if you choose to update/regenerate the documentation!


Configuration Steps
    Before proceeding, ensure that you modify the following paths in the scripts:

For both 01.sh and 02.sh scripts:
    Replace all instances of:
    "/home/kanamori/PycharmProjects/PySASF/docs"
    "/home/kanamori/Pycharmori/PySASF/docs/conf.py"
    with the path to your own 'docs' folder (the one containing this file) within the PySASF main directory and the 'docs/conf.py' file, respectively.

For the 02.sh script only:
    Update the following paths:
    "/home/kanamori/PycharmProjects/PySASF/docs/index.rst"
    "/home/kanamori/PycharmProjects/PySASF/docs/_build/html/index.html"
    to reflect the location of your 'docs/index.rst' and 'docs/_build/html/index.html' files, respectively.
    This manual configuration is necessary for the scripts to function correctly, and automation for this process is planned for future updates.


Reading the Documentation
To simply read the documentation, follow these steps:

    Ensure that no important files are placed within the "PySASF/docs" folder.
    Open a terminal and navigate to the location of the docs folder within the PySASF main directory (i.e., PySASF/docs).
    Execute the following command (without quotes):
    "./02.sh"
    The HTML documentation file should automatically open in your browser. If it does not open immediately, you can find it at docs/_build/html/index.html and open it manually.
    You can now browse the package documentation.


Generating or Updating the Code Documentation
To generate or update the code documentation, please follow these steps:

    Ensure that no important files are placed within the "PySASF/docs" folder.
    Open a terminal and navigate to the location of the docs folder within the PySASF main directory (i.e., PySASF/docs).
    Execute the following command (without quotes):
    "./01.sh"
    If there are any files in the docs folder other than the scripts 01.sh and 02.sh, you will be prompted to delete all files within the docs folder except for these scripts. Please remember: DO NOT PLACE ANY IMPORTANT FILES WITHIN THE "PySASF/docs" FOLDER!
    The HTML documentation file should automatically open in your browser. If it does not open immediately, you can find it at docs/_build/html/index.html and open it manually.
    You can now browse the updated package documentation.


Thank you for using the PySASF package! If you have any questions or need further assistance, please refer to the documentation or reach out for support at https://github.com/tiagoburiol/PySASF
