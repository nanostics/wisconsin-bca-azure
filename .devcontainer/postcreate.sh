# POST CREATE
# Reference (very good blog!!): https://www.sckaiser.com/blog/2023/01/30/conda-codespaces.html


# initialize conda in the container
conda init

# after conda initializes, we can activate the environment
# this makes sure we have our custom env activated in our new terminals
# I find this is a better UX than vscode's "python.terminal.activateEnvironment" setting
echo 'conda activate env' >> ~/.bashrc
