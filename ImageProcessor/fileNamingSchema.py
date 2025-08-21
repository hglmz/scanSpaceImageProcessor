"""
File Naming Schema Module

This module handles custom file naming schema for export operations.
Supports various placeholders for flexible file and folder naming.

Schema Placeholders:
- [r]: Root folder name
- [s]: Sub folder name (if exists)  
- [e]: File name extension (auto-appended if missing)
- [oc]: Original file name without numbers
- [c]: Custom name from custom input
- [o]: Original file name
- [n]: Image Number (padded)
- [n4]: Image Number with 4-digit padding
- [/]: New folder layer

Advanced Separator Patterns:
- [oc-"sep"]: Original filename before separator (e.g., [oc-"_"] from "CAM01_IMG001" = "CAM01")
- [oc+"sep"]: Original filename after separator (e.g., [oc+"_"] from "CAM01_IMG001" = "IMG")
- [r-"sep"]: Root folder name before separator
- [r+"sep"]: Root folder name after separator  
- [o-"sep"]: Original full filename before separator
- [o+"sep"]: Original full filename after separator
- [s-"sep"]: Sub folder name before separator
- [s+"sep"]: Sub folder name after separator

Example Schema: [r]/[s]/[r]_[s]_[n4][e]
Input: ./chineseVase05/crossPolarized/IMG_001.NEF
Output: chineseVase05/crossPolarized/chineseVase05_crossPolarized_0001.jpg

Advanced Separator Examples:
- [oc-"_"][n4] from "CAM01_IMG001.NEF" = "CAM010001.jpg" (auto-extension)
- [oc+"_"][n4][e] from "CAM01_IMG001.NEF" = "IMG0001.jpg"
- [r-"_"]/[r+"_"]_[n4] from root "Project_2024" = "Project/2024_0001.jpg"
- [s+"Polarized"][n4] from subfolder "crossPolarized" = "cross0001.jpg"
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class FileNamingSchema:
    """
    Parser and processor for file naming schemas with custom placeholders.
    """
    
    def __init__(self):
        self.placeholders = {
            'r': 'Root folder name',
            's': 'Sub folder name (if exists)', 
            'e': 'File name extension',
            'oc': 'Original file name without numbers',
            'c': 'Custom name from custom input',
            'o': 'Original file name',
            'n': 'Image Number',
            '/': 'New folder layer'
        }
        
        # Advanced placeholders that support parameters
        self.advanced_placeholders = {
            'oc-': 'Original filename before separator',
            'oc+': 'Original filename after separator',
            'r-': 'Root folder name before separator',
            'r+': 'Root folder name after separator',
            'o-': 'Original filename before separator',
            'o+': 'Original filename after separator',
            's-': 'Sub folder name before separator',
            's+': 'Sub folder name after separator'
        }
    
    def parse_schema(self, schema: str, context: Dict[str, str]) -> Tuple[str, str]:
        """
        Parse a naming schema string and return the output directory and filename.
        
        Args:
            schema: Schema string like "[r]/[s]/[r]_[s]_[n4][e]" or "[oc-\"_\"][n4][e]"
            context: Dictionary containing values for placeholders
            
        Returns:
            Tuple of (output_directory, filename)
        """
        if not schema:
            return "", ""
        
        # Parse numbered padding for [n] placeholder
        def replace_numbered_n(match):
            padding = match.group(1) if match.group(1) else ""
            if padding.isdigit():
                pad_length = int(padding)
                image_num = context.get('n', '1')
                return str(image_num).zfill(pad_length)
            else:
                return context.get('n', '1')
        
        # Parse advanced patterns with separators for [oc], [r], [o], [s]
        def replace_advanced_separator(match):
            placeholder = match.group(1)  # r, o, oc, s
            operation = match.group(2)    # either '-' or '+'
            separator = match.group(3)    # the separator string
            
            # Get the appropriate value from context
            value = context.get(placeholder, '')
            
            if operation == '-':
                # Return part before separator
                if separator in value:
                    return value.split(separator)[0]
                else:
                    return value  # If separator not found, return whole string
            elif operation == '+':
                # Return part after separator
                if separator in value:
                    parts = value.split(separator)
                    return separator.join(parts[1:])  # Join all parts after first separator
                else:
                    return ""  # If separator not found, return empty string
            
            return value
        
        # First, handle all advanced separator patterns [r-"sep"], [o-"sep"], [oc-"sep"], [s-"sep"]
        result = re.sub(r'\[(r|o|oc|s)([+-])"([^"]+)"\]', replace_advanced_separator, schema)
        
        # Then, handle numbered [n] patterns like [n4]
        result = re.sub(r'\[n(\d*)\]', replace_numbered_n, result)
        
        # Replace all other standard placeholders
        for key, value in context.items():
            if key != 'n':  # Already handled above
                placeholder = f'[{key}]'
                result = result.replace(placeholder, str(value))
        
        # Check if extension was included in schema
        has_extension = '[e]' in schema
        
        # Split into directory parts and filename
        parts = result.split('/')
        
        # Find the last part that doesn't contain a file extension
        filename = parts[-1] if parts else ""
        directory_parts = parts[:-1] if len(parts) > 1 else []
        
        # If the last part doesn't have an extension, it might be part of the directory
        if filename and not any(filename.endswith(ext) for ext in ['.jpg', '.png', '.tiff', '.exr']):
            # If no extension placeholder was used, append extension to filename
            if not has_extension and filename:
                filename += context.get('e', '.jpg')
            else:
                directory_parts.append(filename)
                filename = ""
        
        # If we have no filename but no extension was specified, create one
        if not filename and not has_extension:
            filename = f"image{context.get('e', '.jpg')}"
        
        output_directory = '/'.join(directory_parts) if directory_parts else ""
        
        return output_directory, filename
    
    def build_context(self, input_path: str, root_folder: str = "", 
                     custom_name: str = "", image_number: int = 1,
                     output_extension: str = ".jpg", group_name: str = "") -> Dict[str, str]:
        """
        Build context dictionary from input file path and parameters.
        
        Args:
            input_path: Full path to input file
            root_folder: Root folder name (if not derived from path)
            custom_name: Custom name for [c] placeholder
            image_number: Sequential image number
            output_extension: Output file extension (with dot)
            
        Returns:
            Dictionary containing all placeholder values
        """
        path_obj = Path(input_path)

        filename = os.path.basename(path_obj.name)
        filename_no_ext = os.path.splitext(filename)[0]
        
        # Extract original filename without trailing numbers
        filename_no_numbers = re.sub(r'_?\d+$', '', filename_no_ext)

        # Ensure all parameters are strings, not None
        root_folder = root_folder if root_folder is not None else ""
        custom_name = custom_name if custom_name is not None else ""
        group_name = group_name if group_name is not None else ""
        output_extension = output_extension if output_extension is not None else ".jpg"
        
        root_name = os.path.basename(root_folder) if root_folder else ""
        
        context = {
            'r': root_name,
            's': group_name,
            'e': output_extension,
            'oc': filename_no_numbers,
            'c': custom_name,
            'o': filename_no_ext,
            'n': str(image_number)
        }
        
        return context
    
    def validate_schema(self, schema: str) -> Tuple[bool, List[str]]:
        """
        Validate a naming schema and return any errors.
        
        Args:
            schema: Schema string to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if not schema:
            errors.append("Schema cannot be empty")
            return False, errors
        
        # Check for unmatched brackets
        open_brackets = schema.count('[')
        close_brackets = schema.count(']')
        if open_brackets != close_brackets:
            errors.append("Unmatched brackets in schema")
        
        # Find all placeholders, including advanced ones
        placeholders = re.findall(r'\[([^\]]+)\]', schema)
        
        # Check for unknown placeholders
        known_placeholders = set(self.placeholders.keys())
        for placeholder in placeholders:
            # Handle numbered n patterns like n4
            if placeholder.startswith('n') and placeholder[1:].isdigit():
                continue
            elif placeholder == 'n':
                continue
            # Handle advanced separator patterns like oc-"_", r+".", s-"test", o+"sep"
            elif re.match(r'(r|o|oc|s)[+-]"[^"]+"', placeholder):
                continue
            elif placeholder not in known_placeholders:
                errors.append(f"Unknown placeholder: [{placeholder}]")
        
        # Validate advanced separator patterns for all supported placeholders
        advanced_patterns = re.findall(r'\[(r|o|oc|s)([+-])"([^"]+)"\]', schema)
        for placeholder, operation, separator in advanced_patterns:
            if not separator:
                errors.append(f"Empty separator in [{placeholder}{operation}\"\"]")
            elif len(separator) > 10:
                errors.append(f"Separator too long in [{placeholder}{operation}\"{separator}\"] (max 10 chars)")
        
        # Check for invalid characters in Windows file names (excluding quotes in placeholders)
        # First, temporarily replace quoted sections to avoid false positives
        temp_schema = re.sub(r'"[^"]*"', '""', schema)
        invalid_chars = ['<', '>', ':', '|', '?', '*']
        for char in invalid_chars:
            if char in temp_schema:
                errors.append(f"Invalid character '{char}' in schema")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def preview_output(self, schema: str, input_path: str, 
                      custom_name: str = "", image_number: int = 1,
                      output_extension: str = ".jpg", root_folder: str = "") -> str:
        """
        Preview the output path that would be generated from a schema.
        
        Args:
            schema: Schema string
            input_path: Sample input file path
            custom_name: Custom name for preview
            image_number: Image number for preview
            output_extension: Output extension for preview
            root_folder: Root folder override
            
        Returns:
            Preview output path string
        """
        context = self.build_context(input_path, root_folder, custom_name, 
                                   image_number, output_extension)
        
        directory, filename = self.parse_schema(schema, context)
        
        if directory and filename:
            return f"{directory}/{filename}"
        elif filename:
            return filename
        elif directory:
            return f"{directory}/[filename_needed]"
        else:
            return "[invalid_schema]"
    
    def get_placeholder_help(self) -> str:
        """
        Get help text explaining all available placeholders.
        
        Returns:
            Multi-line help text string
        """
        help_text = "Available placeholders:\n"
        for key, description in self.placeholders.items():
            help_text += f"  [{key}]: {description}\n"
        
        help_text += "\nAdvanced separator patterns:\n"
        help_text += "  [oc-\"sep\"]: Original filename before separator\n"
        help_text += "    Example: [oc-\"_\"] from 'CAM01_IMG001' = 'CAM01'\n"
        help_text += "  [oc+\"sep\"]: Original filename after separator\n" 
        help_text += "    Example: [oc+\"_\"] from 'CAM01_IMG001' = 'IMG'\n"
        help_text += "  [r-\"sep\"]: Root folder name before separator\n"
        help_text += "  [r+\"sep\"]: Root folder name after separator\n"
        help_text += "  [o-\"sep\"]: Original full filename before separator\n"
        help_text += "  [o+\"sep\"]: Original full filename after separator\n"
        help_text += "  [s-\"sep\"]: Sub folder name before separator\n"
        help_text += "  [s+\"sep\"]: Sub folder name after separator\n"
        
        help_text += "\nNumber padding:\n"
        help_text += "  [n4]: Image number with 4-digit padding (0001, 0002, etc.)\n" 
        help_text += "  [n2]: Image number with 2-digit padding (01, 02, etc.)\n"
        
        help_text += "\nExamples:\n"
        help_text += "  [r]/[s]/[r]_[s]_[n4][e]\n"
        help_text += "  [oc-\"_\"][n4][e]  # CAM01_IMG001 -> CAM010001.jpg\n"
        help_text += "  [oc+\"_\"][n4][e]  # CAM01_IMG001 -> IMG0001.jpg\n"
        help_text += "  [r-\"_\"]/[r+\"_\"]_[n4][e]  # Split root folder name\n"
        help_text += "  [s-\"Polarized\"]/[o+\"_\"][n4][e]  # Split subfolder and filename"
        
        return help_text


# Helper function for easy integration
def apply_naming_schema(schema: str, input_path: str, output_base_dir: str,
                       custom_name: str = "", image_number: int = 1, 
                       output_extension: str = ".jpg", root_folder: str = "", group_name: str = "") -> str:
    """
    Apply a naming schema to generate a full output path.
    
    Args:
        schema: Naming schema string
        input_path: Input file path  
        output_base_dir: Base output directory
        custom_name: Custom name for [c] placeholder
        image_number: Sequential image number
        output_extension: Output file extension
        root_folder: Root folder name override
        
    Returns:
        Full output file path
    """
    parser = FileNamingSchema()
    
    # Validate schema first
    is_valid, errors = parser.validate_schema(schema)
    if not is_valid:
        raise ValueError(f"Invalid schema: {'; '.join(errors)}")
    
    # Build context and parse
    context = parser.build_context(input_path, root_folder, custom_name,
                                 image_number, output_extension, group_name)
    
    directory, filename = parser.parse_schema(schema, context)
    
    # Combine with base output directory
    if directory:
        full_dir = os.path.join(output_base_dir, directory)
    else:
        full_dir = output_base_dir
        
    if not filename:
        # Fallback filename if schema doesn't produce one
        filename = f"{Path(input_path).stem}_{image_number:04d}{output_extension}"
    
    return os.path.join(full_dir, filename)