
def base_LLM_sampling(sampler_wrapper, input_ids, MAX_NEW_TOKENS, cot: bool = False):
    out = sampler_wrapper.generate(input_ids, MAX_NEW_TOKENS, use_cache=True)

    gen_ids = out.sequences[:, input_ids.shape[1]:].squeeze().detach().cpu()
    completion = sampler_wrapper.tokenizer.decode(gen_ids, skip_special_tokens=True)

    return completion


def low_temperature_sampling(sampler_wrapper, input_ids, MAX_NEW_TOKENS, cot: bool = False):
    out = sampler_wrapper.generate(input_ids, MAX_NEW_TOKENS, use_cache=True)


    gen_ids = out.sequences[:, input_ids.shape[1]:].squeeze().detach().cpu()
    completion = sampler_wrapper.tokenizer.decode(gen_ids, skip_special_tokens=True)

    return completion
