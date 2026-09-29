"""Emit a byte-parity test for ifreflex's rwkv_jev prompt against jev_like.

Compares the rendered prefix and per-question field lead against
1cyberlangke1/rwkv-jev-like (or XingQiPan/rwkv-jev) src/jev_like. Run:

    SEMIF_SRC=... python3 scripts/parity/gen_rwkv_test.py   # see run.sh

The generated C++ compares lengths + an FNV-1a hash so no raw special-token
characters need to be embedded.
"""
import json
import os
import sys

RWKV_SRC = os.environ.get('RWKV_SRC', '/tmp/opencode/port/rwkv-jev-like/src')
sys.path.insert(0, RWKV_SRC)

from jev_like.prompt import build_prefill, build_field_lead  # noqa: E402
from jev_like.primitives import Choice, Noul, Score  # noqa: E402

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


state = "Where is my package? I ordered it last week and it still hasn't arrived."
crit = {
    "billing_question": "Asks about a charge, invoice or payment",
    "cancel_order": "Wants to cancel an order",
    "change_address": "Wants to change the delivery address",
    "report_damage": "Received an item that is broken or damaged",
    "track_order": "Wants to know where an order is or when it arrives",
}
q = Choice("Which intent does the user's message express?", crit)
prefix = build_prefill(state, {"decision": q})
lead = build_field_lead("decision", True)

# question JSON for the C++ side (same request the prompt was built from)
qjson = json.dumps({
    "decision": {"type": "choice",
                 "instructions": "Which intent does the user's message express?",
                 "criteria": crit}
}, ensure_ascii=False)

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
L.append('  json st = %s;' % cxx(state))
L.append('  json qs = json::parse(%s);' % cxx(qjson))
L.append('  rwkv_request rr = build_rwkv_request(st, qs);')
L.append('  if (rr.groups.empty()){printf("FAIL no groups\\n");return 1;}')
L.append('  eq("rwkv.prefix", rr.groups[0].prefix, %d, "%016x");' % (len(prefix), fnv(prefix)))
L.append('  if (rr.groups[0].branches.empty()){printf("FAIL no branch\\n");return 1;}')
L.append('  eq("rwkv.lead", rr.groups[0].branches[0].lead, %d, "%016x");' % (len(lead), fnv(lead)))
# candidate texts join
cands = "".join('billing_question"' for _ in [])
L.append('  {std::string s; for(auto&c:rr.groups[0].branches[0].candidates)s+=c[0]+"|";'
         ' eq("rwkv.candidates", s, %d, "%016x");}' % (
    len("billing_question\"|cancel_order\"|change_address\"|report_damage\"|track_order\"|"),
    fnv("billing_question\"|cancel_order\"|change_address\"|report_damage\"|track_order\"|")))
L.append('  printf("FAILS=%d\\n",fails); return fails?1:0;')
L.append('}')
print('\n'.join(L))
