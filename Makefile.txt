ifeq ($(OS)), Windows_NT)

INSTALL: requirements.txt
		pip install -r requirements.txt


RUN:
		python app.py

else

INSTALL: requirements.txt
		pip3 install -r requirements.txt


RUN:
		python3 app.py
endif