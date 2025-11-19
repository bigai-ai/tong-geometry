#!/bin/bash

# set the host details in .ssh and make "StrictHostKeyChecking no" and "UserKnownHostsFile /dev/null" with identity file log in
# prepare hosts.txt files that list the corresponding servers
# parallel-scp -h hosts.txt ./geometry-gen-scale.zip /root/
# parallel-ssh -h hosts.txt -t 600 -i -I < launch.sh
# to kill all sessions: parallel-ssh -i -h hosts.txt "screen -ls | grep 'Detached' | awk '{print \$1}' | xargs -I {} screen -S {} -X quit"
# to show lines in a file: parallel-ssh -h hosts.txt -i "awk 'NR>=<LINENO> && NR<=<LINENO>' <file>"
# to compress data: parallel-ssh -h hosts.txt -t 600 -i "cd geometry-gen-scale && screen -dmS compress bash -c 'zip -rq data_\$(hostname).zip ./data'; exec bash"
# to unzip certain data: unzip -q data_search-00.zip "data/problems/*.path" -d ./paths
# to count certain files in zip file: unzip -l data_search-00.zip | grep "path" | wc -l

sudo DEBIAN_FRONTEND=noninteractive apt update
# sudo DEBIAN_FRONTEND=noninteractive apt upgrade -y
sudo DEBIAN_FRONTEND=noninteractive apt install -y zip unzip python3-virtualenv screen git

unzip geometry-gen-scale.zip
cd geometry-gen-scale
virtualenv ./venv --python=python3.10
source venv/bin/activate
pip install -r requirements.txt
python setup.py develop

IFS='-' # delimiter must be '-'
read -ra parts <<< "$(hostname)"
SERVER_ID_STR="${parts[-1]}"
SERVER_ID_INT=$((10#${SERVER_ID_STR}))

if [[ $SERVER_ID_INT == 26 ]]; then
	PRIOR_FLAG="--tree"
else
	PRIOR_FLAG="--augment"
fi

echo "source venv/bin/activate && python scripts/priority_mirror_sym.py --cpus-per-node 384 --ck-func ordered --total-samples 10000000000 --seed 20240722 --nodes 27 --node-id $SERVER_ID_INT $PRIOR_FLAG; exec bash"

# Initialize a screen session and run the command
screen -dmS run bash -c "source venv/bin/activate && python scripts/priority_mirror_sym.py --cpus-per-node 384 --ck-func ordered --total-samples 10000000000 --seed 20240722 --nodes 27 --node-id $SERVER_ID_INT $PRIOR_FLAG; exec bash"

