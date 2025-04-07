model_name=vipergpt # custom name for model
split=val # choose from [val, test]
mode=descriptive # choose from [reasoning, descriptive]
openai_key= # OpenAI API key for scoring (e.g., used in src/evaluate.py)

### Query GPT-4o to grade responses ###
python src/evaluate.py \
    --model_name $model_name \
    --split $split \
    --mode $mode \
    --api_key $openai_key

### Get statistics for the model performance ###
python src/get_stats.py \
    --model_name $model_name \
    --split $split 
