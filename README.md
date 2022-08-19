# WSI
### semantic word sense induction projekt
## Construct Corpus 
download wikipedia dump file
```bash
wget https://dumps.wikimedia.org/enwiki/20180220/enwiki-20180220-pages-articles.xml.bz2
```
extracted article,contatains title, section names and plain text section contents, in json-line format
```bash
python -m gensim.scripts.segment_wiki -f enwiki-20180220-pages-articles.xml.bz2 -o enwiki-20180220.json.gz
```


