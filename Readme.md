This project aims at implementing the idea behind what is called a "visual search system". This system is very similar and maybe it is infact a recommender system only.
So the basic idea is, the user inputs an image and the machine(system) outputs a set of similar images from a data set.
There will be many versions of this project and this readme is both instructions for devs/students for cloning this repo and at the same time a kind off diary for me.

Clone the repo : `git clone https://github.com/utk-avi/Visual_Search_System.git` or fork it whatever you like.

Install dependencies: [Bash] `pip install -r requirements.txt`

main.py is the application which currently serves only one significant POST request(POST/upload_image). A user can upload a image based on that.

Take a bunch of different images and store it in a folder(name it dataset or change dataset from image.py to whatever the folder name is) and add the folder to the local cloned repo.

Reload the server : `python -m uvicorn main:app --reload`

Test it from terminal or swaggerUI interface.

Suggest ideas, technical or philosophical to improve this system, thanks!!!
