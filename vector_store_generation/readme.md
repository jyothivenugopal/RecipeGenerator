# to run the script to generate vector store!
cd /Users/charl/Documents/RecipeGenerator/vector_store_generation/
RECIPE_FILE=/Users/charl/.cache/kagglehub/datasets/kaggle/recipe-ingredients-dataset/versions/1/test.json
python generate_vector_store.py \
    ./storage/test_recipe \
    test_recipe \
    $RECIPE_FILE 

python ../agent.py \
    ./storage/test_recipe \
    "I have egg and cheese. What should I make"