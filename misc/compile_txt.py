from sys import argv

## User should supply name of file to save as, then list of files 
## to compile name of file they should be compiled into
## (Note: only first line of each file is used)
## Example :
##
##     python compile_txt.py save_here file1 file2 file3 ...
##

def main(files, name) :
    fil = open(name, "w+")
    for i in files :
        temp_fil = open(i, "r+")
        temp_txt = temp_fil.readline().split(",")[0]
        fil.write(temp_txt + "\n")
        temp_fil.close()
    fil.close()

if __name__ == "__main__" :
    argv.pop(0)
    name = argv.pop(0)
    main(argv, name)
