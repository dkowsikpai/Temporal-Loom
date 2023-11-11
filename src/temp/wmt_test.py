import gzip
import hashlib

with gzip.open('./data/temp/time_stratified_validation.gz', 'rb') as gz_file:
  for line in gz_file:
    date, unsplit_text = line.decode('utf-8').strip().split('\t')
    docid = hashlib.sha256(unsplit_text.encode('utf-8')).hexdigest()
    print(docid, (date, unsplit_text))
    break