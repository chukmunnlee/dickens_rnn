def read_and_tokenize(f):
   for line in f:
      yield line.strip().split(' ')

def tokenize_line(line):
   return line.strip().split(' ')

