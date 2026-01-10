#!/bin/bash
echo "Job started at $(date)"
sleep 3
echo "Job completed at $(date)" > output.txt
ls -la output.txt
exit 0
