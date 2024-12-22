#!/bin/bash
# Path to the conf.py file
directory="/home/kanamori/PycharmProjects/PySASF/docs"
CONF_FILE="/home/kanamori/PycharmProjects/PySASF/docs/conf.py"

LINE_NUMBER_1=8
LINE_NUMBER_2=23
LINE_NUMBER_3=38

NEW_LINES_1="import os\nimport sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path('..', 'pysasf').resolve()))\nsys.path.insert(0, os.path.abspath('..'))"
NEW_LINES_2="extensions = [\n'sphinx.ext.autodoc',\n'sphinx.ext.viewcode',\n'sphinx.ext.napoleon'\n]"
NEW_LINES_3="html_theme = 'bizstyle'"
# Specify the names of the script files to keep
script_name1="01.sh"
script_name2="02.sh"

# Check if the conf.py file exists
if [ -f "$CONF_FILE" ]; then
    read -p "Are you sure you want to delete this Sphinx Configuration and make a new one? (y/n) " answer

    if [ "$answer" = "y" ]; then
        # Delete all files except the specified script files
        find "$directory" -type f ! -name "$script_name1" ! -name "$script_name2" -exec rm -f {} +
    else
        exit 0
    fi    
fi

cd /home/kanamori/PycharmProjects/PySASF/docs
sphinx-quickstart <<EOF
n
PySASF
Tiago Buriol
0.5
en
EOF

# Add new lines to conf.py
sed -i "${LINE_NUMBER_1}a $NEW_LINES_1" "$CONF_FILE"
sed -i "${LINE_NUMBER_2}a $NEW_LINES_2" "$CONF_FILE"
sed -i "${LINE_NUMBER_3}a $NEW_LINES_3" "$CONF_FILE"
wait
cd
# Append the command to run 02.sh at the end of conf.py
cd /home/kanamori/PycharmProjects/PySASF/docs
./02.sh
<<EOF


