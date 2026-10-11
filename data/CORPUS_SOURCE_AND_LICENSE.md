# Sample source and attribution

`paper128_test.txt` contains the first 128-byte packet of the former test corpus,
an ASCII excerpt from [Robert Boulter](https://en.wikipedia.org/wiki/Robert_Boulter)
by Wikipedia contributors. The article's history provides contributor details.

The text comes from the official test split of
[WikiText-2 raw](https://huggingface.co/datasets/Salesforce/wikitext), revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, file
`wikitext-2-raw-v1/test-00000-of-00001.parquet`. Dataset creators: Stephen Merity,
Caiming Xiong, James Bradbury, and Richard Socher.

Preparation included punctuation transliteration, NFKD normalization,
combining-mark removal, detokenization, heading-marker removal, and replacement
of remaining non-ASCII characters with spaces. This distribution retains only
the first packet; no text was added or padded.

The prepared text adaptation is distributed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
([legal text](https://creativecommons.org/licenses/by-sa/4.0/legalcode)).
Credit Wikipedia contributors and the WikiText creators when redistributing.
Upstream metadata also lists CC BY-SA 3.0 and GFDL. These terms apply to the
sample text, not the Python code. No endorsement is implied.
