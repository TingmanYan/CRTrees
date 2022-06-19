#!/bin/bash
# Copyright (c) 2016, David Stutz
# Contact: david.stutz@rwth-aachen.de, davidstutz.de
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Example of evaluating W.
# Supposed to be run from within examples/.

echo "CRTREES on dataset: $1"

SUPERPIXELS=("200" "300" "400" "600" "800" "1000" "1200" "1400" "1600" "1800" "2000" "2400" "2800" "3200" "3600" "4000" "4600" "5200")

rm -rf ../superpixel-benchmark/output/$1/crtrees*

for SUPERPIXEL in "${SUPERPIXELS[@]}"
do
    ./build/CRTREES_img_seq ../superpixel-benchmark/data/$1/images/test/ --superpixels $SUPERPIXEL -o ../superpixel-benchmark/output/$1/crtrees/$SUPERPIXEL -w -l 3 -n 8 -g 0.0
    ../superpixel-benchmark/bin/eval_summary_cli ../superpixel-benchmark/output/$1/crtrees/$SUPERPIXEL ../superpixel-benchmark/data/$1/images/test ../superpixel-benchmark/data/$1/csv_groundTruth/test --append-file ../superpixel-benchmark/output/$1/crtrees.csv
    find ../superpixel-benchmark/output/$1/crtrees/$SUPERPIXEL -type f -name '*[^summary|correlation|results].csv' -delete
done

../superpixel-benchmark/bin/eval_average_cli ../superpixel-benchmark/output/$1/crtrees.csv -o ../superpixel-benchmark/output/$1/crtrees_average.csv
