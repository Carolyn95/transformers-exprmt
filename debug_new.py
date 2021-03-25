from transformers import (AdamW, T5ForConditionalGeneration, T5Tokenizer,
                          get_linear_schedule_with_warmup)

model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')
max_len = 32
input_ids = tokenizer(['The <extra_id_0> walks in <extra_id_1> park'],
                      max_length=max_len,
                      padding='max_length',
                      is_split_into_words=False,
                      return_tensors='pt').input_ids
print(input_ids)
labels = tokenizer(['<extra_id_0> cute dog <extra_id_1> the <extra_id_2>'],
                   max_length=max_len,
                   padding='max_length',
                   is_split_into_words=False,
                   return_tensors='pt').input_ids
print(labels)
# the forward function automatically creates the correct decoder_input_ids
loss = model(input_ids=input_ids, labels=labels).loss
print(loss)
