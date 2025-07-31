def analyze_log(log_content):
    """
    Analyzes the log file and extracts key information including:
    - Loaded plugins
    - Processing steps with timestamps
    - DE00 values at each step
    - Skipped steps
    - Total processing time
    """
    import re
    from collections import defaultdict
    
    analysis = {
        'plugins': [],
        'steps': [],
        'de00_values': {},
        'skipped_steps': [],
        'processing_time': None,
        'raw_decoding_time': None,
        'flat_field_time': None
    }
    
    lines = log_content.split('\n')
    time_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
    plugin_pattern = re.compile(r'Plugin (\w+) loaded successfully')
    de00_pattern = re.compile(r'Current DE00 at step (\w+\.\w+): (\d+\.\d+)')
    time_taken_pattern = re.compile(r'Time taken: (\d+\.\d+) seconds')
    skipped_pattern = re.compile(r'The step (\w+\.\w+) didn\'t improve DE00')
    
    # Track timestamps for specific events
    raw_decode_start = None
    flat_field_start = None
    
    for line in lines:
        # Check for plugin loading
        plugin_match = plugin_pattern.search(line)
        if plugin_match:
            analysis['plugins'].append(plugin_match.group(1))
        
        # Check for DE00 values
        de00_match = de00_pattern.search(line)
        if de00_match:
            step = de00_match.group(1)
            value = float(de00_match.group(2))
            analysis['de00_values'][step] = value
            analysis['steps'].append(step)
            
            # Get timestamp for the step
            time_match = time_pattern.search(line)
            if time_match:
                analysis['steps'].append(f"{step} at {time_match.group(1)}")
        
        # Check for skipped steps
        skipped_match = skipped_pattern.search(line)
        if skipped_match:
            analysis['skipped_steps'].append(skipped_match.group(1))
        
        # Check for processing time
        time_taken_match = time_taken_pattern.search(line)
        if time_taken_match:
            analysis['processing_time'] = float(time_taken_match.group(1))
        
        # Track RAW decoding time
        if "Decoding RAW file..." in line:
            time_match = time_pattern.search(line)
            if time_match:
                raw_decode_start = time_match.group(1)
        elif "RAW file decoded successfully" in line and raw_decode_start:
            time_match = time_pattern.search(line)
            if time_match:
                # Simple time difference calculation (for demo purposes)
                # In production, you'd want to use datetime parsing
                start_sec = int(raw_decode_start.split(':')[-1])
                end_sec = int(time_match.group(1).split(':')[-1])
                analysis['raw_decoding_time'] = end_sec - start_sec
        
        # Track flat field correction time
        if "Applying flat-field correction..." in line:
            time_match = time_pattern.search(line)
            if time_match:
                flat_field_start = time_match.group(1)
        elif "Flat-field correction applied" in line and flat_field_start:
            time_match = time_pattern.search(line)
            if time_match:
                # Simple time difference calculation
                start_sec = int(flat_field_start.split(':')[-1])
                end_sec = int(time_match.group(1).split(':')[-1])
                analysis['flat_field_time'] = end_sec - start_sec
    
    return analysis

def extract_polynomial_fitting_values(log_content):
    """
    Extracts all information related to POLYNOMIAL_FITTING_CORRECTOR from the log
    including DE00 values, timestamps, and whether it was skipped.
    """
    import re
    
    result = {
        'de00_value': None,
        'timestamp': None,
        'was_skipped': False,
        'time_taken': None
    }
    lines = log_content.split('\n')
    time_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
    de00_pattern = re.compile(r'Current DE00 at step AnalysisSteps.POLYNOMIAL_FITTING_CORRECTOR: (\d+\.\d+)')
    time_taken_pattern = re.compile(r'Time taken: (\d+\.\d+) seconds')
    skipped_pattern = re.compile(r'The step AnalysisSteps.POLYNOMIAL_FITTING_CORRECTOR didn\'t improve DE00')
    
    for line in lines:
        if "POLYNOMIAL_FITTING_CORRECTOR" in line:
            # Check for DE00 value
            de00_match = de00_pattern.search(line)
            if de00_match:
                result['de00_value'] = float(de00_match.group(1))
                
                # Get timestamp
                time_match = time_pattern.search(line)
                if time_match:
                    result['timestamp'] = time_match.group(1)
            
            # Check if it was skipped
            if skipped_pattern.search(line):
                result['was_skipped'] = True
            
            # Check for time taken
            time_taken_match = time_taken_pattern.search(line)
            if time_taken_match:
                result['time_taken'] = float(time_taken_match.group(1))
    
    return result