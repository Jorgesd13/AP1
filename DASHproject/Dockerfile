FROM python:3.10

#ODBC setup
SHELL ["/bin/bash", "--login", "-c"]
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
RUN curl https://packages.microsoft.com/config/ubuntu/18.04/prod.list > /etc/apt/sources.list.d/mssql-release.list
RUN apt-get update -y
RUN ACCEPT_EULA=Y apt-get install -y msodbcsql17 unixodbc-dev mssql-tools
RUN apt-get update -y

#Python env setup
RUN mkdir -p /home/project/dash
WORKDIR /home/project/dash
COPY requirements.txt /home/project/dash
COPY DigiPythonTools /home/project/dash/DigiPythonTools
RUN pip install --upgrade pip --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn

COPY . /home/project/dash