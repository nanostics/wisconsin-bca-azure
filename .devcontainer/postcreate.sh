# POST CREATE
# Reference (very good blog!!): https://www.sckaiser.com/blog/2023/01/30/conda-codespaces.html


# initialize conda in the container
# somehow this always fails with a `TypeError: memoryview: a bytes-like object is required, not 'str'`
# but conda is still inittialized. I just ignore the error with `|| true` lol
# the "yes N" is to skip the interactivity of conda init when it fails
yes N | conda init || true 

# after conda initializes, we can activate the environment
# this makes sure we have our custom env activated in our new terminals
echo 'conda activate env' >> ~/.bashrc

# bash colours: https://stackoverflow.com/a/5947802
# Tell users that the conda error is a-ok!!
GREEN='\033[0;32m'
NC='\033[0m' # No Color
printf "\nIf there is a conda error, ${GREEN}DO NOT WORRY!! EVERYTHING IS OKAY!!${NC}\nSee .devcontainer/postcreate.sh for more info.\n"
