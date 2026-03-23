#!/usr/bin/env bash

input_file="data/IRP/out.log"
output_file="data/IRP/good_out.log"

awk '
    /IRP_MJ_WRITE/ {
        # Split the line into an array using spaces or tabs as delimiters
        split($0, arr, /[ \t]+/);

        # Start rebuilding the line with the first 8 fields (these are fine as is)
        $0 = arr[1] "\t" arr[2] "\t" arr[3] "\t" arr[4] "                       \t" arr[5] "                      \t" arr[6] "\t" arr[7] " " arr[8];

        # Add a tab between "myexe.exe" and "PreSize"
        $0 = $0 "\t" arr[9];
        $0 = $0 " " arr[10];
        $0 = $0 "\t" arr[11];
        $0 = $0 " " arr[12];


        # Append the rest of the fields starting from the 10th position onward
        for (i=13; i<=length(arr); i++) {
            $0 = $0 "\t" arr[i];
        }

        # Output the corrected line
        print $0;
    }

    # For lines that dont contain IRP_MJ_WRITE, just print them as they are
    !/IRP_MJ_WRITE/ { print }
' "$input_file" > "$output_file"

echo "File has been processed. The fixed file is saved as '$output_file'."