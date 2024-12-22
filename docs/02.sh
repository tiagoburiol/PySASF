#!/bin/bash

LINE_NUMBER_1=13
indexRSTfile="/home/kanamori/PycharmProjects/PySASF/docs/index.rst"

# Define the new line with the desired indentation (4 spaces)
NEW_LINES_1="   modules"

# Change to the project directory
cd /home/kanamori/PycharmProjects/PySASF

# Generate the Sphinx API documentation
sphinx-apidoc -o docs pysasf/
sleep 2
# Check if the line already exists in the file
if grep -Fxq "$NEW_LINES_1" "$indexRSTfile"; then
    echo "The line already exists in $indexRSTfile. No changes made."
else
    # Create a temporary file to hold the modified content
    temp_file=$(mktemp)

    # Use awk to insert the new line at the specified line number
    awk -v new_line="$NEW_LINES_1" -v line_num="$LINE_NUMBER_1" 'NR==line_num {print new_line} 1' "$indexRSTfile" > "$temp_file"

    # Move the temporary file back to the original file
    mv "$temp_file" "$indexRSTfile"

    echo "Added the line to $indexRSTfile."
fi
echo "teste01"
# Change to the docs directory and build the HTML
cd
cd /home/kanamori/PycharmProjects/PySASF/docs
make clean html
cd
# Open the generated HTML file
xdg-open /home/kanamori/PycharmProjects/PySASF/docs/_build/html/index.html
