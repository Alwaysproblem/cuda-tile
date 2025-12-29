#!/bin/bash

set -eo pipefail

rsync -avP example example.tilebc 5090:~/Desktop/

ssh 5090 'bash -s' << EOF
#!/bin/bash
set -eo pipefail

cd ~/Desktop/
./example

rm -rf example.tilebc example
EOF
