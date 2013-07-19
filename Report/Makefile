# Change the following to whatever you like. Target will have '.ps' or '.pdf'
# appended as need be. Tex should be the tex file WITHOUT the '.tex' suffix.
TEX=Fijee
TARGET=Fijee



# You probably don't need to change anything below the following line:
.PHONY: clean debug gvshow pdfshow acroshow dvishow
LATEX_ARGS=--interaction batchmode
LATEX=latex



ifeq ($(PARAM), pdf)
LATEX_ARGS = "\def\PDF{}"
#LATEX_ARGS = "\newcommand{\KINDLE}{}" 
TARGET=Fijee.pdf
else
LATEX_ARGS += 
TARGET=Fijee.ps
endif

#
# make -> make ps
# make ps -> make ps
# make pdf -> make pdf
# make PARAM=kindle -> make kindle format
#

all: $(TARGET)
	@echo $(PARAM)

Fijee.ps: Fijee.dvi
	dvips Fijee -o Fijee.ps

Fijee.dvi: Fijee.tex
	$(LATEX) $(LATEX_ARGS) Fijee
	bibtex Fijee
	makeindex -s Fijee.ist Fijee
	$(LATEX) $(LATEX_ARGS) Fijee
	$(LATEX) $(LATEX_ARGS) Fijee
#	@mv Fijee.dvi Fijee.dvi

Fijee.pdf: Fijee.tex
	pdflatex $(LATEX_ARGS) Fijee

clean: 
	find . -name \*.pdf -exec rm {} \;
	find . -name \*.toc -exec rm {} \;
	find . -name \*.ind -exec rm {} \;
	find . -name \*.aux -exec rm {} \;
	find . -name \*.ps -exec rm {} \;
	find . -name \*.log -exec rm {} \;
	find . -name \*.idx -exec rm {} \;
	find . -name \*.bbl -exec rm {} \;
	find . -name \*.blg -exec rm {} \;
	find . -name \*.dvi -exec rm {} \;
	find . -name \*.ilg -exec rm {} \;
	find . -name \*.maf -exec rm {} \;
	find . -name \*~ -exec rm {} \;
	find . -name \*.mtc\* -exec rm {} \;

debug:
	latex Fijee

ps: Fijee.ps
	gv Fijee.ps

pdf:  Fijee.pdf
	xpdf Fijee.pdf

kindle: Fijee.pdf 
	xpdf Fijee.pdf

dvi: Fijee.dvi
	xdvi Fijee.dvi