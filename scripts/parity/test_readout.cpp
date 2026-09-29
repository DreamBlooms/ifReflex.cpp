#include <cmath>
#include <cstdio>
#include <vector>
#include <string>
#include "ifreflex/readout.hpp"
using namespace ifreflex;

static double maxdiff = 0.0;
static void chk(const char* tag, double got, double want) {
    double d = std::fabs(got - want);
    if (d > maxdiff) maxdiff = d;
    std::printf("%s got=%.12f want=%.12f diff=%.2e %s\n", tag, got, want, d, d < 1e-6 ? "OK" : "MISMATCH");
}

int main() {
    std::vector<std::vector<float>> L = {
        {0.5f,-1.25f,2.0f,0.0f},
        {1.75f,0.25f,-0.5f,3.25f,-2.0f},
        {-0.1f,0.2f,-0.3f}};
    std::vector<std::vector<double>> SW = {
        {0.159693550032,0.027750577933,0.715696837782,0.096859034253},
        {0.171416732873,0.038248243058,0.018067190722,0.768236498188,0.004031335159},
        {0.315597833313,0.426012514949,0.258389651738}};
    std::vector<double> SWT[] = {{0.099055180788,0.008130944379,0.844322237242,0.048491637591},
        {0.227719034408,0.083773151122,0.050810984616,0.619004513227,0.018692316626},
        {0.327056247378,0.373704545444,0.299239207177}};
    double temps[] = {0.7,1.5,2.25};
    double CONF[] = {0.381122502911,0.549877661905,0.019526049087};

    for (int i=0;i<3;i++) {
        auto p = softmax(L[i], 1.0);
        for (size_t j=0;j<p.size();j++) chk("softmax", p[j], SW[i][j]);
        auto pt = softmax(L[i], temps[i]);
        for (size_t j=0;j<pt.size();j++) chk("softmaxT", pt[j], SWT[i][j]);
        chk("confidence", confidence(softmax(L[i],1.0)), CONF[i]);
    }

    question_kind KS[] = {question_kind::noul, question_kind::choice, question_kind::score};
    int ST[] = {17,200,1};
    std::vector<std::vector<double>> FEAT = {
        {1.0,1.0,0.0,0.0,1.386294361120,0.283321334406,0.618877497089,0.15},
        {1.0,0.0,1.0,0.0,1.609437912434,0.529831736655,0.450122338095,0.15},
        {1.0,0.0,0.0,1.0,1.098612288668,0.0,0.980473950913,0.03}};
    double HTMP[] = {2.703244336160,2.763372358053,4.711311227202};
    calibration cal; cal.head = {0.03,-0.11,0.07,0.02,0.4,-0.25,1.1,-0.6};
    for (int i=0;i<3;i++) {
        auto f = head_features(KS[i], L[i], ST[i]);
        for (size_t j=0;j<f.size();j++) chk("feature", f[j], FEAT[i][j]);
        chk("headT", cal.t(KS[i], L[i], ST[i]), HTMP[i]);
    }

    // merge: choice, two perms
    std::vector<std::vector<std::string>> keys = {
        {"payments","account","other"},{"other","payments","account"}};
    std::vector<std::vector<float>> lg = {{1.0f,0.0f,-1.0f},{-1.0f,1.0f,0.0f}};
    auto m = merge_branches(question_kind::choice, keys, lg, calibration{}, 10);
    chk("merge.0", m["payments"], 0.665240955775);
    chk("merge.1", m["account"],  0.244728471055);
    chk("merge.2", m["other"],    0.090030573170);

    // answers
    auto a = to_answer(question_kind::choice,
        {{"account",0.3},{"payments",0.5},{"other",0.2}}, json());
    chk("choice.conf", a.at("confidence").get<double>(), 0.062769000000);
    if (a.at("choice").get<std::string>() != "payments") std::printf("choice.argmax MISMATCH\n");
    else std::printf("choice.argmax OK\n");

    auto s = to_answer(question_kind::score,
        {{"0",0.1},{"1",0.7},{"2",0.2}}, {{"0","low"},{"1","mid"},{"2","high"}});
    chk("score.value", s.at("score").get<double>(), 1.100000000000);
    chk("score.conf", s.at("confidence").get<double>(), 0.270153000000);

    auto n = to_answer(question_kind::noul, {{"false",0.28},{"true",0.72}}, json());
    chk("noul.value", n.at("noul").get<double>(), 0.720000000000);

    std::printf("\nMAXDIFF=%.3e\n", maxdiff);
    return maxdiff > 1e-6 ? 1 : 0;
}
