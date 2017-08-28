.PHONY: all
all : mea.py schemes.py

.PHONY: test
test: test_mpdata.py test_upwind.py
	sage -python -m pytest

%.py: %.sage
	sage $<
	mv $<.py $@

.PHONY: clean
clean:
	rm -f *.py *.pyc
