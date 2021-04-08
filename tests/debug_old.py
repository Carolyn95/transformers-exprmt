from transformers import (AdamW, T5ForConditionalGeneration, T5Tokenizer,
                          get_linear_schedule_with_warmup)

model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')
max_len = 32
input_ids = tokenizer.batch_encode_plus(
    ['The <extra_id_0> walks in <extra_id_1> park'],
    max_length=max_len,
    extra_ids=100,
    is_pretokenized=True,
    pad_to_max_length=True,
    return_tensors='pt').input_ids
print(input_ids)
label_text = ['<extra_id_0> cute dog <extra_id_1> the <extra_id_2>']
label_text = tokenizer.prepare_for_tokenization(label_text)
labels = tokenizer.batch_encode_plus(label_text,
                                     max_length=max_len,
                                     extra_ids=100,
                                     is_pretokenized=True,
                                     pad_to_max_length=True,
                                     return_tensors='pt').input_ids
print(labels)
# the forward function automatically creates the correct decoder_input_ids
loss = model(input_ids=input_ids, lm_labels=labels)[0]
print(loss)
# print(tokenizer.convert_ids_to_tokens(3155))