import json, sys
import os
sys.path.insert(0, os.environ.get('SEMIF_SRC', '/tmp/opencode/port/SemIf-OpenJev/src'))
from semif_phase1.core import direct_messages, DIRECT_SYSTEM

FNV = 1469598103934665603
def fnv(s):
    h = FNV
    for b in s.encode():
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h

def cxx(s):
    out=[]
    for ch in s:
        o=ord(ch)
        if ch=='"': out.append('\\"')
        elif ch=='\\': out.append('\\\\')
        elif ch=='\n': out.append('\\n')
        elif o<32 or o>126: out.append('\\x%02x'%o)
        else: out.append(ch)
    return '"'+''.join(out)+'"'

state={"ticket":"Refund denied.","plan":"pro"}
row={"id":"r1","state":state,"question":"Which team?",
     "options":[{"id":"payments","description":"payouts"},{"id":"account","description":""}]}
msgs=direct_messages(row)
user=json.loads(msgs[1]["content"])
cont={"criterion":user["criterion"],"options":user["options"]}
user_full=json.dumps(user, ensure_ascii=False)
prefix_body='{"evidence": '+json.dumps(state,ensure_ascii=False)+', '
branch_body=json.dumps(cont, ensure_ascii=False)[1:]

A="<"+"|"+"im_start"+"|"+">"
B="<"+"|"+"im_end"+"|"+">"
tail = B + "\n" + A + "assistant" + "\n"
expected_prefix = A + "system\n" + DIRECT_SYSTEM + B + "\n" + A + "user\n" + prefix_body
expected_branch = branch_body + tail
expected_full = expected_prefix + expected_branch

L=[]
L.append('#include <cstdio>')
L.append('#include <string>')
L.append('#include "ifreflex/prompt.hpp"')
L.append('using namespace ifreflex;')
L.append('static int fails=0;')
L.append('static unsigned long long fnv(const std::string&s){unsigned long long h=1469598103934665603ULL;for(char c:s){h^=(unsigned char)h;' if False else 'static unsigned long long fnv(const std::string&s){unsigned long long h=1469598103934665603ULL;for(char c:s){h^=(unsigned char)c;h*=1099511628211ULL;}return h;}')
L.append('static std::string hexv(unsigned long long v){char b[33];snprintf(b,33,"%016llx",v);return b;}')
L.append('static void eq(const char* t,const std::string& g,size_t gw,const char* wh){std::string gh=hexv(fnv(g));if(g.size()==gw && gh==wh)printf("OK   %s\\n",t);else{printf("FAIL %s got_len=%zu want_len=%zu got=%s want=%s\\n",t,g.size(),gw,gh.c_str(),wh);fails++;}}')
L.append('static json ST = json::parse(%s);'%cxx(json.dumps(state,ensure_ascii=False)))
L.append('int main(){')
L.append('  prompt_format f; f.style=prompt_style::semif;')
L.append('  eq("semif.sys", system_prompt(f), %d, "%016x");'%(len(DIRECT_SYSTEM), fnv(DIRECT_SYSTEM)))
L.append('  eq("semif.prefix", render_state_prefix(f, ST), %d, "%016x");'%(len(expected_prefix), fnv(expected_prefix)))
inst=json.dumps(row["question"],ensure_ascii=False)
crit=json.dumps({"payments":"payouts","account":""},ensure_ascii=False)
call='build_branches(f,"r1",question_kind::choice, json::parse(%s), json::parse(%s), 1)'%(cxx(inst),cxx(crit))
L.append('  eq("semif.body", %s[0].text, %d, "%016x");'%(call, len(expected_branch), fnv(expected_branch)))
L.append('  eq("semif.full", render_state_prefix(f, ST) + %s[0].text, %d, "%016x");'%(call, len(expected_full), fnv(expected_full)))
L.append('  eq("semif.n", std::to_string(%s.size()), 1, "%016x");'%(call, fnv("1")))
L.append('  printf("FAILS=%d\\n",fails); return fails?1:0;')
L.append('}')
print('\n'.join(L))
