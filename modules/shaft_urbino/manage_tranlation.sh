#!/bin/bash

# This script manages the automation of the localization process for the application.
# It uses Babel to handle the POT, PO, and MO files necessary for the application's translation.

# How to generate the POT file and PO files:
# - To generate the POT file for the first time or update it with new strings, run './manage_translation.sh extract'.
# - To create or update the PO files based on the updated POT file, run './manage_translation.sh update'.
# - To compile the MO files from the updated PO files, run './manage_translation.sh compile'.
# - To perform all steps in sequence, use './manage_translation.sh all'.

# Define the base path where your project is located
BASE_PATH="./"

# Define the application name as used in gettext
DOMAIN="co.co.a"

# Define the translations directory
LOCALE_DIR="$BASE_PATH/locales"

# Define the Babel configuration file
BABEL_CONFIG="$BASE_PATH/babel.cfg"

# Output POT file
POT_FILE="$LOCALE_DIR/messages.pot"

# Configuration file
CONFIG_FILE="$BASE_PATH/config.ini"

# The sequence of operations is as follows:
# 1. Extraction of strings into the POT file.
# 2. Updating the PO files with the new strings from the POT file.
# 3. Compilation of the PO files into MO files.

# Check and create necessary folders if they don't exist
function check_create_dirs() {
    # Read the list of languages from the configuration file
    languages=$(awk -F'=' '/supported/ {print $2}' $CONFIG_FILE | tr -d ' ')

    # Create directories for each language, excluding English
    for lang in ${languages//,/ }
    do
        if [ "$lang" != "en" ]; then  # Skip creation for English language
            if [ ! -d "$LOCALE_DIR/$lang/LC_MESSAGES" ]; then
                echo "Creating directory $LOCALE_DIR/$lang/LC_MESSAGES..."
                mkdir -p "$LOCALE_DIR/$lang/LC_MESSAGES"
            fi
        fi
    done
}


# Extract strings into POT files
function extract_strings() {
    echo "Extracting strings..."
    pybabel extract -F $BABEL_CONFIG -o $POT_FILE $BASE_PATH
}

# Update PO files with new strings from POT file
function update_po_files() {
    echo "Updating PO files..."

    # Leggere la lista delle lingue dal file di configurazione
    languages=$(awk -F'=' '/supported/ {print $2}' $CONFIG_FILE | tr -d ' ')

    # Aggiornare o creare file PO per ogni lingua, escluso l'inglese
    for lang in ${languages//,/ }
    do
        if [ "$lang" != "en" ]; then  # Saltare l'inglese
            local po_file="$LOCALE_DIR/$lang/LC_MESSAGES/$DOMAIN.po"

            # Controllare se il file PO esiste
            if [ -f "$po_file" ]; then
                # Aggiornare il file PO esistente
                echo "Updating $lang PO file..."
                pybabel update -i $POT_FILE -d $LOCALE_DIR -l $lang -D $DOMAIN
            else
                # Creare un nuovo file PO se non esiste
                echo "Creating new $lang PO file..."
                pybabel init -i $POT_FILE -d $LOCALE_DIR -l $lang -D $DOMAIN
            fi
        fi
    done
}


# Compile PO files into MO files
function compile_mo_files() {
    echo "Compiling MO files..."
    pybabel compile -d $LOCALE_DIR -D $DOMAIN
}

# Check arguments to decide which function to execute
case "$1" in
    extract)
        check_create_dirs
        extract_strings
        ;;
    update)
        check_create_dirs
        update_po_files
        ;;
    compile)
        check_create_dirs
        compile_mo_files
        ;;
    all)
        check_create_dirs
        extract_strings
        update_po_files
        compile_mo_files
        ;;
    *)
        echo "Usage: $0 {extract|update|compile|all}"
        exit 1
        ;;
esac

