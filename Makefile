VENV?=venv
PIP?=$(VENV)/bin/pip
PYTH?=$(VENV)/bin/python
PROJ?=src/
MAX_COMPLEXITY?=5


.PHONY : install
.PHONY : virtualenv
.PHONY : update
.PHONY : pull
.PHONY : clean-py
.PHONY : clean
.PHONY : py-version-tests
.PHONY : fix-style
.PHONY : tests
.PHONY : check-style
.PHONY : checks


# install requirements
install: virtualenv
	$(PIP) install -r requirements.txt

virtualenv:
	python3 -m virtualenv $(VENV) --system-site-packages
	$(PIP) install --upgrade pip


update: pull install clean


pull:
	git pull


# remove python bytecode files
clean-py:
	find $(PROJ) -name "*.py[cod]" -exec rm -f {} \;
	find $(PROJ) -name "__pycache__" -type d -exec rm -f {} \;


# clean out unwanted files
clean: clean-py


# run tests in different versions of python
py-version-tests:
	tox

# automatically make python files pep 8-compliant
# (see tox.ini for autopep8 constraints)
fix-style:
	autopep8 -r --in-place --aggressive --aggressive $(MRLN)


# run all official tests
tests: py-version-tests


# run code style checks
check-style:
	-$(PYTH) -m flake8 --max-complexity $(MAX_COMPLEXITY) $(MRLN)
	-pylint $(MRLN)


# run all checks
checks: check-style
