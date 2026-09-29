"""Byte-parity test for ifreflex's granite4 template against the GGUF's own Jinja
chat template (rendered with jinja2, transformers-compatible).

For a reflex_markdown request with an empty branch, the full prompt equals:
    render_state_prefix(fmt, state) + assistant_tail(fmt)
and the Granite template renders the same system+user pair. This compares the two.

    GRANITE_GGUF=/path/granite-4.0-h-tiny-Q8_0.gguf python3 gen_granite_test.py
"""
import json
import os
import struct
import sys

SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

# reflex markdown system prompt (must match ifreflex system_prompt()).
REFLEX_SYS = ("You are a System One decision model. You read the State and answer "
              "each Question by choosing exactly one of the listed options. You never "
              "explain. You answer with the single option label only.")


def read_meta(path):
    f = open(path, 'rb')
    assert f.read(4) == b'GGUF'
    ver, nt, nkv = struct.unpack('<IQQ', f.read(20))

    def gs():
        l = struct.unpack('<Q', f.read(8))[0]
        return f.read(l).decode('utf-8', 'replace')

    kv = {}
    for _ in range(nkv):
        k = gs()
        t = struct.unpack('<I', f.read(4))[0]
        if t == 8:
            kv[k] = gs()
        elif t == 9:
            et = struct.unpack('<I', f.read(4))[0]
            n = struct.unpack('<Q', f.read(8))[0]
            if et == 8:
                kv[k] = [gs() for _ in range(n)]
            else:
                f.read(SIZES[et] * n)
        elif t in SIZES:
            raw = f.read(SIZES[t])
            kv[k] = struct.unpack({0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I',
                                   5: '<i', 6: '<f', 7: '<?', 10: '<Q',
                                   11: '<q', 12: '<d'}[t], raw)[0]
    return kv


FNV = 1469598103934665603


def fnv(s):
    h = FNV
    for b in s.encode():
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def cxx(s):
    out = []
    for ch in s:
        o = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == '\\':
            out.append('\\\\')
        elif ch == '\n':
            out.append('\\n')
        elif o < 32 or o > 126:
            out.append('\\x%02x' % o)
        else:
            out.append(ch)
    return '"' + ''.join(out) + '"'


def main():
    gguf = os.environ['GRANITE_GGUF']
    kv = read_meta(gguf)
    tpl = kv['tokenizer.chat_template']

    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from jinja2 import StrictUndefined
    env = ImmutableSandboxedEnvironment(undefined=StrictUndefined)
    env.filters['tojson'] = lambda x, **k: json.dumps(x, **k)

    state = "hello world"
    user_body = "# Evidence\n" + state + "\n\n"          # reflex_markdown body, empty branch
    msgs = [{"role": "system", "content": REFLEX_SYS},
            {"role": "user", "content": user_body}]
    rendered = env.from_string(tpl).render(
        messages=msgs, tools=None, documents=None,
        add_generation_prompt=True, bos_token='', eos_token='')

    L = []
    L.append('#include <cstdio>')
    L.append('#include <string>')
    L.append('#include "ifreflex/prompt.hpp"')
    L.append('using namespace ifreflex;')
    L.append('static int fails=0;')
    L.append('static unsigned long long fnv(const std::string&s){unsigned long long h=1469598103934665603ULL;for(char c:s){h^=(unsigned char)c;h*=1099511628211ULL;}return h;}')
    L.append('static std::string hexv(unsigned long long v){char b[33];snprintf(b,33,"%016llx",v);return b;}')
    L.append('static void eq(const char* t,const std::string& g,size_t gw,const char* wh){std::string gh=hexv(fnv(g));if(g.size()==gw&&gh==wh)printf("OK   %s\\n",t);else{printf("FAIL %s got_len=%zu want_len=%zu got=%s want=%s\\n",t,g.size(),gw,gh.c_str(),wh);fails++;}}')
    L.append('int main(){')
    L.append('  prompt_format f; f.style = prompt_style::reflex_markdown;')
    L.append('  f.template_kind = template_style::granite4;')
    L.append('  json st = %s;' % cxx(state))
    L.append('  std::string p = render_state_prefix(f, st) + prompt_suffix(f);')
    L.append('  eq("granite.full", p, %d, "%016x");' % (len(rendered), fnv(rendered)))
    L.append('  printf("FAILS=%d\\n",fails); return fails?1:0;')
    L.append('}')
    print('\n'.join(L))
    sys.stderr.write("python-rendered len=%d\n" % len(rendered))


if __name__ == '__main__':
    main()
