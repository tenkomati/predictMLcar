predictorflask
==============================

A webapp developed in Flask for predicting car prices. it uses Mercadolibre's API and four different machine learning regression models.

LAST CHANGES:

-Comparative analisis for same car on different years for the 20% top models of Mercadolibre, from 2000 to 2023 DONE (there is an issue with the average price because its calculated from all the trims and using all the values, even the outliers)

Next I will add this:
-taking the trim into account to calculate the statistics.
-Vehicle specifications: I want to bring information about the selected car, trim levels and all the relevant information.
-Historical sales data: historical sales data of the vehicle (i need to get this via a public record or something)
-photo of the car model:
-most related problems of this model:
-map of the actual offer of this model:

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   └── raw            <- csv generated files
    ├── static             <- generated graphics with matplotlib 
    ├── templates          <- html code
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pipreqs > requirements.txt`
    ├── Dockerfile         <- Docker file
    ├── app.py             <- THE CODE
    ├── functions.py       <- Functions used in the code
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
