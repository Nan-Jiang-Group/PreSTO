import re
from typing import Optional


def extract_boxed(text: str) -> Optional[str]:
    pattern = r'\\boxed\s*\{'
    
    for match in re.finditer(pattern, text):
        start = match.end()
        brace_count = 1
        end = start
        
        while end < len(text) and brace_count > 0:
            if text[end] == '{':
                brace_count += 1
            elif text[end] == '}':
                brace_count -= 1
            end += 1
            
        if brace_count == 0:
            content = text[start:end-1]
            return content.strip()
            
    return None


def extract_last_boxed(text: str) -> Optional[str]:
    pattern = r'\\boxed\s*\{'
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None
    last_match = matches[-1]
    start = last_match.end()
    brace_count = 1
    end = start
    
    while end < len(text) and brace_count > 0:
        if text[end] == '{':
            brace_count += 1
        elif text[end] == '}':
            brace_count -= 1
        end += 1
        
    if brace_count == 0:
        content = text[start:end-1]
        return content.strip()
        
    return None


def extract_numeric_answer(text: str) -> Optional[str]:
    patterns = [
        r'(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*([+-]?\d+(?:\.\d+)?(?:/\d+)?)',
        r'(?:therefore|thus|hence|so)[,\s]+(?:the\s+)?(?:answer\s+is\s+)?([+-]?\d+(?:\.\d+)?(?:/\d+)?)',
        r'=\s*([+-]?\d+(?:\.\d+)?(?:/\d+)?)\s*$',
        r'\*\*([+-]?\d+(?:\.\d+)?(?:/\d+)?)\*\*\s*$',  # Markdown bold
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1)
            
    numbers = re.findall(r'[+-]?\d+(?:\.\d+)?(?:/\d+)?', text)
    if numbers:
        return numbers[-1]
        
    return None


def extract_latex_expression(text: str) -> Optional[str]:
    inline_pattern = r'\$([^$]+)\$'
    display_pattern = r'\$\$([^$]+)\$\$'
    
    display_matches = re.findall(display_pattern, text)
    if display_matches:
        return display_matches[-1].strip()
    inline_matches = re.findall(inline_pattern, text)
    if inline_matches:
        return inline_matches[-1].strip()
        
    return None


def extract_answer(text: str) -> str:
    text = text.strip()
    
    if not text:
        return ""
    
    boxed = extract_last_boxed(text)
    if boxed:
        return boxed
        
    answer_patterns = [
        r'(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*(.+?)(?:\.|$)',
        r'(?:therefore|thus|hence|so)[,\s]+(?:the\s+)?(?:answer\s+is\s+)?(.+?)(?:\.|$)',
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            answer = match.group(1).strip()
            answer = re.sub(r'\s+', ' ', answer)
            if len(answer) < 100:  # Sanity check
                return answer
                
    latex = extract_latex_expression(text)
    if latex:
        return latex
        
    numeric = extract_numeric_answer(text)
    if numeric:
        return numeric
        
    return ""


def normalize_answer(answer: str) -> str:
    if not answer:
        return ""
        
    answer = answer.strip()
    
    answer = answer.replace('$', '')
    
    boxed_pattern = r'\\boxed\s*\{(.+)\}'
    match = re.match(boxed_pattern, answer, re.DOTALL)
    if match:
        answer = match.group(1).strip()
    
    latex_with_content = [
        (r'\\text\{([^}]*)\}', r'\1'),
        (r'\\textbf\{([^}]*)\}', r'\1'),
        (r'\\textit\{([^}]*)\}', r'\1'),
        (r'\\mathrm\{([^}]*)\}', r'\1'),
        (r'\\mathbf\{([^}]*)\}', r'\1'),
    ]
    
    for pattern, repl in latex_with_content:
        answer = re.sub(pattern, repl, answer)
    
    for cmd in ['\\left', '\\right', '\\displaystyle', '\\;', '\\,', '\\!', '\\ ']:
        answer = answer.replace(cmd, '')
            
    frac_pattern = r'\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}'
    for _ in range(5):
        new_answer = re.sub(frac_pattern, r'\1/\2', answer)
        if new_answer == answer:
            break
        answer = new_answer
        
    for frac_cmd in ['dfrac', 'tfrac', 'cfrac']:
        pattern = r'\\' + frac_cmd + r'\s*\{([^{}]*)\}\s*\{([^{}]*)\}'
        for _ in range(5):
            new_answer = re.sub(pattern, r'\1/\2', answer)
            if new_answer == answer:
                break
            answer = new_answer
            
    sqrt_pattern = r'\\sqrt\s*\{([^{}]*)\}'
    answer = re.sub(sqrt_pattern, r'sqrt(\1)', answer)
    answer = re.sub(r'\\[a-zA-Z]+', '', answer)
    answer = ' '.join(answer.split())
    answer = answer.lower()
    answer = answer.replace('×', '*')
    answer = answer.replace('·', '*')
    answer = answer.replace('÷', '/')
    answer = answer.replace('−', '-')
    answer = answer.replace('–', '-')
    
    answer = answer.rstrip('.,;')
    
    answer = re.sub(r'\s+', ' ', answer)
    answer = answer.strip()
    
    answer = answer.rstrip('.,;')
    
    try:
        if re.match(r'^[+-]?\d+(?:\.\d+)?$', answer):
            val = float(answer)
            if val == int(val):
                return str(int(val))
            return answer
        elif re.match(r'^[+-]?\d+/\d+$', answer):
            parts = answer.split('/')
            val = float(parts[0]) / float(parts[1])
            if val == int(val):
                return str(int(val))
    except:
        pass
        
    return answer.strip()


def answers_match(pred: str, gold: str) -> bool:
    norm_pred = normalize_answer(pred)
    norm_gold = normalize_answer(gold)
    
    if norm_pred == norm_gold:
        return True
        
    try:
        val_pred = eval_numeric(norm_pred)
        val_gold = eval_numeric(norm_gold)
        
        if val_pred is not None and val_gold is not None:
            return abs(val_pred - val_gold) < 1e-6
    except:
        pass
        
    return False


def eval_numeric(s: str) -> Optional[float]:
    if not s:
        return None
        
    if '/' in s and not any(c in s for c in ['sqrt', 'pi', 'e']):
        try:
            parts = s.split('/')
            if len(parts) == 2:
                num = float(parts[0].strip('() '))
                den = float(parts[1].strip('() '))
                return num / den
        except:
            pass
            
    s_eval = s.replace('pi', str(3.141592653589793))
    try:
        return float(s_eval)
    except:
        pass
        
    return None
