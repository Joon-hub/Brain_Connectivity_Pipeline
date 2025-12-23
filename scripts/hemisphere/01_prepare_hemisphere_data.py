#!/usr/bin/env python3
"""
Hemisphere Data Split Pipeline
===============================

Transforms raw connectivity data (wide format with upper triangle) into 
hemisphere-specific datasets (wide format with upper triangle per hemisphere).

Input Format:
    - CSV with columns: subject, region1~region2, region3~region4, ...
    - Each row = one subject
    - Connectivity values in upper triangle format

Output Format:
    - Two CSV files per input: LH_*.csv and RH_*.csv
    - Each file: subject column + upper triangle connections for that hemisphere
    - Shape: (n_subjects, 6671) for 116 regions per hemisphere
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import sys


class HemisphereSplitter:
    """Handles the transformation and splitting of connectivity data by hemisphere."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.regions_ordered = []
        self.connectivity_map = {}
        
    def log(self, message: str, level: str = "INFO"):
        """Print formatted log message."""
        if self.verbose:
            print(f"[{level}] {message}")
    
    def print_separator(self, title: str = ""):
        """Print a formatted separator line."""
        if self.verbose:
            print(f"\n{'='*60}")
            if title:
                print(f"{title}")
                print(f"{'='*60}")
    
    def load_data(self, filepath: Path) -> pd.DataFrame:
        """Load raw connectivity data from CSV."""
        self.print_separator("LOADING DATA")
        self.log(f"Reading: {filepath}")
        
        df = pd.read_csv(filepath)
        self.log(f"Shape: {df.shape}")
        self.log(f"Subjects: {df['subject'].nunique()}")
        
        return df
    
    def extract_regions(self, df: pd.DataFrame) -> List[str]:
        """
        Extract unique regions from connectivity column names while preserving order.
        Connectivity columns are formatted as 'region1~region2'.
        """
        self.print_separator("EXTRACTING REGIONS")
        
        # Get all connectivity columns
        connectivity_cols = [col for col in df.columns if '~' in str(col)]
        self.log(f"Connectivity columns found: {len(connectivity_cols)}")
        
        # Extract unique regions in order
        regions_ordered = []
        seen_regions = set()
        
        for col in connectivity_cols:
            region1, region2 = col.split('~')
            
            if region1 not in seen_regions:
                regions_ordered.append(region1)
                seen_regions.add(region1)
            
            if region2 not in seen_regions:
                regions_ordered.append(region2)
                seen_regions.add(region2)
        
        self.log(f"Total unique regions: {len(regions_ordered)}")
        self.log(f"First 5: {regions_ordered[:5]}")
        self.log(f"Last 5: {regions_ordered[-5:]}")
        
        self.regions_ordered = regions_ordered
        return regions_ordered
    
    def build_connectivity_map(self, df: pd.DataFrame) -> Dict:
        """
        Create a mapping of connectivity values for quick lookup. Format: {subject: {(region1, region2): value}}
        """
        self.print_separator("BUILDING CONNECTIVITY MAP")
        
        connectivity_cols = [col for col in df.columns if '~' in str(col)]
        connectivity_map = {}
        
        for idx, row in df.iterrows():
            subject = row['subject']
            connectivity_map[subject] = {}
            
            for col in connectivity_cols:
                region1, region2 = col.split('~')
                connectivity_map[subject][(region1, region2)] = row[col]
        
        self.log(f"Connectivity map created for {len(connectivity_map)} subjects")
        
        self.connectivity_map = connectivity_map
        return connectivity_map
    
    def transform_to_long_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform from wide format (subject × connections) to long format 
        (subject × region × all_regions).
        """
        self.print_separator("TRANSFORMING TO LONG FORMAT")
        
        rows = []
        n_subjects = df['subject'].nunique()
        n_regions = len(self.regions_ordered)
        
        self.log(f"Creating {n_subjects} subjects × {n_regions} regions = {n_subjects * n_regions} rows")
        
        for subject in df['subject']:
            for target_region in self.regions_ordered:
                # Create a row for this subject-region pair
                row_data = {
                    'subject': subject,
                    'region': target_region
                }
                
                # For each of the 232 regions, find connectivity value
                for source_region in self.regions_ordered:
                    if target_region == source_region:
                        # Diagonal: check if self-connection exists, otherwise assign 1
                        if (target_region, source_region) in self.connectivity_map[subject]:
                            row_data[source_region] = self.connectivity_map[subject][(target_region, source_region)]
                        elif (source_region, target_region) in self.connectivity_map[subject]:
                            row_data[source_region] = self.connectivity_map[subject][(source_region, target_region)]
                        else:
                            row_data[source_region] = 1.0
                    else:
                        # Off-diagonal: check both directions
                        if (target_region, source_region) in self.connectivity_map[subject]:
                            row_data[source_region] = self.connectivity_map[subject][(target_region, source_region)]
                        elif (source_region, target_region) in self.connectivity_map[subject]:
                            row_data[source_region] = self.connectivity_map[subject][(source_region, target_region)]
                        else:
                            row_data[source_region] = np.nan
                
                rows.append(row_data)
        
        # Create the dataframe
        df_transformed = pd.DataFrame(rows)
        
        # Ensure column order: subject, region, then all regions in order
        final_columns = ['subject', 'region'] + self.regions_ordered
        df_transformed = df_transformed[final_columns]
        
        self.log(f"Transformed shape: {df_transformed.shape}")
        self.log(f"Expected: ({n_subjects * n_regions}, {n_regions + 2})")
        self.log(f"Missing values: {df_transformed.isna().sum().sum()}")
        
        return df_transformed
    
    def identify_hemispheres(self) -> Tuple[List[str], List[str]]:
        """
        Identify left and right hemisphere regions from the ordered region list.
        Left: LH_ prefix OR -lh suffix
        Right: RH_ prefix OR -rh suffix
        """
        self.print_separator("IDENTIFYING HEMISPHERES")
        
        left_regions = [reg for reg in self.regions_ordered 
                       if reg.startswith('LH_') or reg.endswith('-lh')]
        right_regions = [reg for reg in self.regions_ordered 
                        if reg.startswith('RH_') or reg.endswith('-rh')]
        
        # Detailed breakdown
        left_cortical = [reg for reg in left_regions if reg.startswith('LH_')]
        left_subcortical = [reg for reg in left_regions if reg.endswith('-lh')]
        right_cortical = [reg for reg in right_regions if reg.startswith('RH_')]
        right_subcortical = [reg for reg in right_regions if reg.endswith('-rh')]
        
        self.log(f"LEFT HEMISPHERE: {len(left_regions)} total")
        self.log(f"  Cortical (LH_): {len(left_cortical)}")
        self.log(f"  Subcortical (-lh): {len(left_subcortical)}")
        
        self.log(f"RIGHT HEMISPHERE: {len(right_regions)} total")
        self.log(f"  Cortical (RH_): {len(right_cortical)}")
        self.log(f"  Subcortical (-rh): {len(right_subcortical)}")
        
        self.log(f"\nFirst 5 left: {left_regions[:5]}")
        self.log(f"Last 5 left: {left_regions[-5:]}")
        self.log(f"First 5 right: {right_regions[:5]}")
        self.log(f"Last 5 right: {right_regions[-5:]}")
        
        return left_regions, right_regions
    
    def split_by_hemisphere(self, df_transformed, left_regions, right_regions):
        self.print_separator("SPLITTING BY HEMISPHERE")

        # 1. Strictly define columns: ONLY subject, target region, and the features for THAT hemi
        left_cols = ['subject', 'region'] + left_regions
        right_cols = ['subject', 'region'] + right_regions
        
        # 2. Filter Rows AND Columns simultaneously
        df_left = df_transformed.loc[df_transformed['region'].isin(left_regions), left_cols].copy()
        df_right = df_transformed.loc[df_transformed['region'].isin(right_regions), right_cols].copy()
        
        # 3. Double Check: Print counts to console
        for name, df in [("Left", df_left), ("Right", df_right)]:
            found_wrong = [c for c in df.columns if ('RH_' in c if name=="Left" else 'LH_' in c)]
            if found_wrong:
                print(f"[ERROR] {name} file contains {len(found_wrong)} regions from the wrong hemisphere!")
            else:
                print(f"[SUCCESS] {name} file is pure.")
                
        return df_left, df_right
    
    def create_upper_triangle_columns(self, regions: List[str]) -> List[str]:
        """
        Create upper triangle connectivity column names preserving region order.
        Only includes i < j pairs (no diagonal).
        """
        columns = []
        n = len(regions)
        for i in range(n):
            for j in range(i + 1, n):  # i < j ensures upper triangle
                columns.append(f"{regions[i]}~{regions[j]}")
        return columns
    
    def transform_to_wide_format(self, df_long: pd.DataFrame, 
                                 regions: List[str],
                                 hemisphere: str) -> pd.DataFrame:
        """
        Transform from long format back to wide format with only upper triangle connections.
        Args:
            df_long: Long format dataframe (rows per subject-region)
            regions: Ordered list of regions for this hemisphere
            hemisphere: 'left' or 'right' (for logging)
        """
        self.print_separator(f"TRANSFORMING {hemisphere.upper()} TO WIDE FORMAT")
        
        # Create upper triangle column names
        upper_cols = self.create_upper_triangle_columns(regions)
        self.log(f"Upper triangle connections: {len(upper_cols)}")
        self.log(f"Expected: {len(regions) * (len(regions) - 1) // 2}")
        
        # Create dictionary to store wide format data
        wide_data = {'subject': df_long['subject'].unique()}
        n_subjects = len(wide_data['subject'])
        
        self.log(f"Processing {n_subjects} subjects...")
        
        # For each upper triangle connection
        for idx, conn in enumerate(upper_cols):
            if idx % 1000 == 0 and idx > 0:
                self.log(f"  Progress: {idx}/{len(upper_cols)} connections processed")
            
            region_i, region_j = conn.split('~')
            
            # Get connectivity values for this pair
            values = []
            for subject in wide_data['subject']:
                # Get the row where subject matches and region matches region_i
                row = df_long[(df_long['subject'] == subject) & (df_long['region'] == region_i)]
                if not row.empty:
                    # Get the connectivity value from region_i to region_j
                    value = row[region_j].values[0]
                    values.append(value)
                else:
                    values.append(np.nan)
            
            wide_data[conn] = values
        
        df_wide = pd.DataFrame(wide_data)
        
        self.log(f"Wide format shape: {df_wide.shape}")
        self.log(f"Expected: ({n_subjects}, {len(upper_cols) + 1})")
        self.log(f"Missing values: {df_wide.isna().sum().sum()}")
        
        return df_wide
    
    def save_hemisphere_data(self, df_wide: pd.DataFrame, 
                            output_path: Path,
                            hemisphere: str):
        """Save hemisphere data to CSV."""
        df_wide.to_csv(output_path, index=False)
        self.log(f"✓ Saved: {output_path}")
        self.log(f"  Shape: {df_wide.shape}")
        self.log(f"  First connection: {df_wide.columns[1]}")
        self.log(f"  Last connection: {df_wide.columns[-1]}")
    
    def process_dataset(self, input_path: Path, output_dir: Path) -> Tuple[Path, Path]:
        """
        Main processing pipeline for a single dataset.
        
        Returns:
            Tuple of (left_output_path, right_output_path)
        """
        self.print_separator(f"PROCESSING: {input_path.name}")
        
        # Step 1: Load data
        df_raw = self.load_data(input_path)
        
        # Step 2: Extract regions
        self.extract_regions(df_raw)
        
        # Step 3: Build connectivity map
        self.build_connectivity_map(df_raw)
        
        # Step 4: Transform to long format
        df_transformed = self.transform_to_long_format(df_raw)
        
        # Step 5: Identify hemispheres
        left_regions, right_regions = self.identify_hemispheres()
        
        # Step 6: Split by hemisphere
        df_left, df_right = self.split_by_hemisphere(df_transformed, left_regions, right_regions)
        
        # Step 7: Transform to wide format
        df_left_wide = self.transform_to_wide_format(df_left, left_regions, "left")
        df_right_wide = self.transform_to_wide_format(df_right, right_regions, "right")
        
        # Step 8: Generate output filenames
        # Extract base name (e.g., "PIOP2_restingstate" -> "LH_PIOP2_RestingState")
        base_name = input_path.stem  # Remove .csv
        
        # Convert to title case and add hemisphere prefix
        parts = base_name.split('_')
        formatted_name = '_'.join([p.capitalize() for p in parts])
        
        left_filename = f"LH_{formatted_name}.csv"
        right_filename = f"RH_{formatted_name}.csv"
        
        left_output = output_dir / left_filename
        right_output = output_dir / right_filename
        
        # Step 9: Save files
        self.print_separator("SAVING FILES")
        self.save_hemisphere_data(df_left_wide, left_output, "left")
        self.save_hemisphere_data(df_right_wide, right_output, "right")
        
        return left_output, right_output


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Split connectivity data into hemisphere-specific datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process both default datasets
    python hemisphere_data_split.py
    
    # Process specific files
    python hemisphere_data_split.py --input data/raw/PIOP2_restingstate.csv
    
    # Specify custom output directory
    python hemisphere_data_split.py --output-dir data/processed/hemispheres/
    
    # Quiet mode
    python hemisphere_data_split.py --quiet
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        nargs='+',
        help='Input CSV file(s) to process. If not specified, processes both PIOP datasets.'
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/raw',
        help='Directory containing input files (default: data/raw)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed/hemispheres',
        help='Directory for output files (default: data/processed/hemispheres)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which files to process
    if args.input:
        input_files = [Path(f) if Path(f).is_absolute() else input_dir / f 
                      for f in args.input]
    else:
        # Default: process both PIOP datasets
        input_files = [
            input_dir / 'PIOP2_restingstate.csv',
            input_dir / 'PIOP1_gstroop.csv'
        ]
    
    # Verify input files exist
    for filepath in input_files:
        if not filepath.exists():
            print(f"[ERROR] File not found: {filepath}")
            sys.exit(1)
    
    # Initialize processor
    splitter = HemisphereSplitter(verbose=not args.quiet)
    
    # Process each dataset
    all_outputs = []
    for input_file in input_files:
        try:
            left_out, right_out = splitter.process_dataset(input_file, output_dir)
            all_outputs.extend([left_out, right_out])
        except Exception as e:
            print(f"[ERROR] Failed to process {input_file}: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Final summary
    if not args.quiet:
        print(f"\n{'='*60}")
        print(f"PIPELINE COMPLETE!")
        print(f"{'='*60}")
        print(f"Processed {len(input_files)} dataset(s)")
        print(f"Generated {len(all_outputs)} output file(s):")
        for out_file in all_outputs:
            print(f"  ✓ {out_file}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()