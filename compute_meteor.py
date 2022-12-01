from nltk.translate.meteor_score import meteor_score
import sys

def readFile(path):
  with open(path) as f:
    text=f.read()
  return text.split('\n')
  
def main(argv):
  if len(argv) != 3:
    print >>sys.stderr, 'usage: %s <source file> <target file>' % argv[0]
    print >>sys.stderr, '    <source file> : Permutation'
    print >>sys.stderr, '    <target file> : Reference'
    sys.exit(1)
  src = readFile(argv[1])
  tgt = readFile(argv[2])
  scores = [meteor_score([y],x) for x,y in zip(src,tgt)]
  print('meteor: {}'.format(sum(scores)/len(scores)))

if __name__ == '__main__':
  main(sys.argv)
