#!/bin/bash
rsync -avz --delete requirements.txt src wsgi.py spyros@eydap-backend.ddns.net:eydap-backend
ssh spyros@eydap-backend.ddns.net 'source ~/eydap-backend/venv/bin/activate && pip install -r ~/eydap-backend/requirements.txt'
ssh spyros@eydap-backend.ddns.net 'sudo supervisorctl reload'
