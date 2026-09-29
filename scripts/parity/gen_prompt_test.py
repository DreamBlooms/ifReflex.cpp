import json, sys
import os
sys.path.insert(0, os.environ.get('REFLEX_SRC', '/tmp/opencode/port/reflex/src'))
from reflex.prompt import PromptFormat, build_branches
from reflex.schema import NoulQuestion, ChoiceQuestion, ScoreQuestion

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
        elif ch == '\t':
            out.append('\\t')
        elif o < 32 or o > 126:
            out.append('\\x%02x' % o)
        else:
            out.append(ch)
    return '"' + ''.join(out) + '"'

def criteria_json(q):
    if isinstance(q, NoulQuestion):
        c = q.criteria
        return {"true": c.true, "false": c.false} if c else None
    return q.criteria

def keystr(k):
    if k is True:
        return "true"
    if k is False:
        return "false"
    return str(k)

state = {"ticket": "Refund denied for order 42.", "plan": "pro"}
qs = [
  ("queue", ChoiceQuestion(type="choice", instructions="Which team?",
     criteria={"payments": "payouts, refunds", "account": None, "other": "everything else"})),
  ("escalate", NoulQuestion(type="noul", instructions="Escalate to a manager?",
     criteria={"true": "Yes, escalate.", "false": "No, do not."})),
  ("urgency", ScoreQuestion(type="score", instructions="How urgent?",
     criteria=["can wait a week", "handled today", "needs attention now"])),
]

L = []
L.append('#include <cstdio>')
L.append('#include <string>')
L.append('#include <vector>')
L.append('#include "ifreflex/prompt.hpp"')
L.append('using namespace ifreflex;')
L.append('static int fails=0;')
L.append('static std::string esc(const std::string& s){std::string o;for(char c:s){unsigned char u=(unsigned char)c;if(c==10)o+="\\n";else if(u<32||c==34){char b[8];snprintf(b,8,"\\\\x%02x",u);o+=b;}else o+=c;}return o;}')
L.append('static void eq(const char* t,const std::string& g,const std::string& w){if(g==w)printf("OK   %s\\n",t);else{printf("FAIL %s\\n  got(%zu):  %s\\n  want(%zu): %s\\n",t,g.size(),esc(g).c_str(),w.size(),esc(w).c_str());fails++;}}')
L.append('static json ST = json::parse(%s);' % cxx(json.dumps(state, ensure_ascii=False)))
L.append('int main(){')

for style in ["markdown", "compact"]:
    fmt = PromptFormat(style=style)
    key = "reflex_markdown" if style == "markdown" else "reflex_compact"
    L.append('  {')
    L.append('    prompt_format f; f.style=prompt_style::%s;' % key)
    L.append('    eq("sys.%s", system_prompt(f), %s);' % (style, cxx(fmt.system_prompt)))
    L.append('    eq("prefix.%s", render_state_prefix(f, ST), %s);' % (style, cxx(fmt.prefix(state))))
    for qid, q in qs:
        cj = criteria_json(q)
        inst = json.dumps(q.instructions, ensure_ascii=False)
        crit = json.dumps(cj, ensure_ascii=False)
        brs = build_branches(qid, q, fmt, permutations=2)
        kindmap = {"choice": "choice", "noul": "noul", "score": "score"}
        b = brs[0]
        call = 'build_branches(f,%s,question_kind::%s, json::parse(%s), json::parse(%s), 2)' % (
            cxx(qid), kindmap[q.type], cxx(inst), cxx(crit))
        L.append('    eq("br.%s.%s.0", %s[0].text, %s);' % (style, qid, call, cxx(b.text)))
        L.append('    eq("brlabels.%s.%s.0", [&](){auto v=%s[0].labels; std::string s; for(auto&x:v)s+=x+","; return s;}(), %s);'
                 % (style, qid, call, cxx(",".join(str(l) for l in b.labels) + ",")))
        L.append('    eq("brkeys.%s.%s.0", [&](){auto v=%s[0].keys; std::string s; for(auto&x:v)s+=x+","; return s;}(), %s);'
                 % (style, qid, call, cxx(",".join(keystr(k) for k in b.keys) + ",")))
    L.append('  }')

L.append('  printf("\\nFAILS=%d\\n",fails); return fails?1:0;')
L.append('}')
print('\n'.join(L))
