import inspect
from datetime import datetime
import subprocess

def debug(msg, noheader=False, sep="\t"):
    """Print to console with debug information."""
    caller = inspect.stack()[1]
    file_name = caller.filename
    ln = caller.lineno
    now = datetime.now()
    time = now.strftime("%m/%d/%Y - %H:%M:%S")
    if noheader:
        print("\t\x1b[94m{}\x1b[0m".format(msg), end="")
        return
    print(
        '\x1b[40m[{}] File "{}", line {}\x1b[0m\n\t\x1b[94m{}\x1b[0m'.format(
            time, file_name, ln, msg
        )
    )

def joren_graphPoj_script():
    return "script_joren/get_func_graph.scala"

def joren_graphJava250_script():
    return "java250/get_func_graph.scala"

def docker_dir():
    pass

def subprocess_cmd(command: str, verbose: int = 0, force_shell: bool = False):
    """Run command line process.
    Example:
    subprocess_cmd('echo a; echo b', verbose=1)
    >>> a
    >>> b
    """
    # singularity = os.getenv("SINGULARITY")
    # if singularity != "true" and not force_shell:
    #     command = f"singularity exec {docker_dir() / 'main.sif'} " + command
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    output = process.communicate()
    if verbose > 1:
        debug(output[0].decode())
        debug(output[1].decode())
    return output

import random
def split_data_shuffle(data_file, output_dir, num_piece=10):
    with open(data_file, "r") as f:
        lines = f.readlines()
        random.shuffle(lines)
        s = len( lines )
        chunk_size = int(s/num_piece)
        chunks=[lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
        for i, data in enumerate( chunks ):
            with open( f"{output_dir}/{i+1}.jsonl", "w" ) as ff:
                    ff.writelines(data)

def watch_subprocess_cmd(command: str, force_shell: bool = False):
    """Run subprocess and monitor output. Used for debugging purposes."""
    #singularity = os.getenv("SINGULARITY")
    #if singularity != "true" and not force_shell:
    #    command = f"singularity exec {docker_dir() / 'main.sif'} " + command
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    # Poll process for new output until finished
    noheader = False
    while True:
        nextline = process.stdout.readline()
        if nextline == b"" and process.poll() is not None:
            break
        debug(nextline.decode(), noheader=noheader)
        noheader = True



def get_networkx_graph(line_data):
    from networkx.drawing import nx_agraph
    import pygraphviz
    import json
    data = json.loads(line_data)
    #json.dump(data, open("tmpgraph.json", "w"), indent=6)

    ast_str_list=json.loads(data["ast"])
    ast_graph = [ nx_agraph.from_agraph(pygraphviz.AGraph(ast_str)) for ast_str in ast_str_list ]

    cdg_str_list=json.loads(data["cdg"])
    cdg_graph = [ nx_agraph.from_agraph(pygraphviz.AGraph(cdg_str)) for cdg_str in cdg_str_list ]

    ddg_str_list=json.loads(data["ddg"])
    ddg_graph = [ nx_agraph.from_agraph(pygraphviz.AGraph(ddg_str)) for ddg_str in ddg_str_list ]

    cfg_str_list=json.loads(data["cfg"])
    cfg_graph = [ nx_agraph.from_agraph(pygraphviz.AGraph(cfg_str)) for cfg_str in cfg_str_list ]
    return {"index":data["index"],"ast":ast_graph, "cdg":cdg_graph, "ddg":ddg_graph, "cfg":cfg_graph}
