#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
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

    // logmean: geometric mean across branches. With two branches and an additive
    // position bias the geometric mean cancels it, so the result is the un-biased
    // distribution rather than the arithmetic mean of the biased ones.
    {
        std::vector<std::vector<std::string>> k2 = {
            {"payments","account","other"}, {"other","payments","account"}};
        std::vector<std::vector<float>> l2 = {{1.0f,0.0f,-1.0f},{-1.0f,1.0f,0.0f}};
        auto lm = merge_branches(question_kind::choice, k2, l2, calibration{}, 10, combine_mode::logmean);
        // Branch 0: softmax([1,0,-1]); branch 1 is the same option scores shifted.
        // geometric mean per option = softmax of the option's mean logit.
        chk("logmean.0", lm["payments"], 0.665240955775);
        chk("logmean.1", lm["account"],  0.244728471055);
        chk("logmean.2", lm["other"],    0.090030573170);

        // A single branch: mean and logmean must agree.
        std::vector<std::vector<std::string>> k1 = {{"a","b"}};
        std::vector<std::vector<float>> l1 = {{0.0f, 1.0f}};
        auto m1 = merge_branches(question_kind::choice, k1, l1, calibration{}, 5, combine_mode::mean);
        auto l1m = merge_branches(question_kind::choice, k1, l1, calibration{}, 5, combine_mode::logmean);
        chk("combine.single.a", l1m["a"], m1["a"]);
        chk("combine.single.b", l1m["b"], m1["b"]);
    }

    // apply_prior: normalize(p / prior^strength). strength 0 is a no-op.
    {
        std::map<std::string, double> p = {{"a",0.6},{"b",0.4}};
        std::map<std::string, double> pr = {{"a",0.8},{"b",0.2}};
        auto none = apply_prior(p, pr, 0.0);
        chk("prior.s0.a", none["a"], 0.6);
        chk("prior.s0.b", none["b"], 0.4);
        auto full = apply_prior(p, pr, 1.0);
        // a: 0.6/0.8=0.75 ; b: 0.4/0.2=2.0 ; sum 2.75
        chk("prior.s1.a", full["a"], 0.75/2.75);
        chk("prior.s1.b", full["b"], 2.0/2.75);
        auto half = apply_prior(p, pr, 0.5);
        const double ra = 0.6/std::sqrt(0.8), rb = 0.4/std::sqrt(0.2), rs = ra+rb;
        chk("prior.s05.a", half["a"], ra/rs);
        chk("prior.s05.b", half["b"], rb/rs);
    }

    // batch_prior: running mean of the option distributions.
    {
        batch_prior bp;
        chk("bp.count0", (double) bp.count(), 0.0);
        if (bp.ready(2)) std::printf("bp.ready MISMATCH\n"); else std::printf("bp.ready OK\n");
        bp.add({{"a",0.5},{"b",0.5}}, {"a","b"});
        bp.add({{"a",1.0},{"b",0.0}}, {"a","b"});
        if (!bp.ready(2)) std::printf("bp.ready2 MISMATCH\n"); else std::printf("bp.ready2 OK\n");
        chk("bp.count", (double) bp.count(), 2.0);
        auto m = bp.mean();
        chk("bp.a", m["a"], 0.75);
        chk("bp.b", m["b"], 0.25);
    }

    // fit_temperature: minimise the per-branch NLL of the restricted softmax over a
    // temperature multiplier. A perfectly separated set wants a cold temperature;
    // a uniform set wants a hot one.
    {
        auto nll_over = [&](double tau, const std::vector<std::vector<float>> & lg,
                            const std::vector<std::vector<double>> & y) {
            double loss = 0.0;
            for (size_t i = 0; i < lg.size(); ++i)
                loss += nll(softmax(lg[i], tau), y[i]);
            return loss;
        };

        // A well-separated set wants a cold temperature: the fit must lower the NLL
        // against the uncalibrated T=1 and stay inside the search bounds.
        std::vector<std::vector<float>> sep = {{5.0f,-5.0f},{6.0f,-6.0f},{4.0f,-4.0f}};
        std::vector<std::vector<double>> ysep = {{1.0,0.0},{1.0,0.0},{1.0,0.0}};
        std::vector<int> st = {10,10,10};
        const double t_sep = fit_temperature(question_kind::choice, sep, ysep, st, calibration{});
        chk("fit.sep.bounded", t_sep >= 1.0/20.0 && t_sep <= 20.0 ? 1.0 : 0.0, 1.0);
        chk("fit.sep.improves", nll_over(t_sep, sep, ysep) < nll_over(1.0, sep, ysep) ? 1.0 : 0.0, 1.0);

        // The fit must be at (or very near) the NLL optimum, not merely better.
        for (double tau : {0.5 * t_sep, 2.0 * t_sep}) {
            chk("fit.sep.optimal", nll_over(t_sep, sep, ysep) <= nll_over(tau, sep, ysep) + 1e-6 ? 1.0 : 0.0, 1.0);
        }

        // An ambiguous set (no margin) also wants a cold fit; the contract is only
        // that the result is bounded and does not raise the NLL.
        std::vector<std::vector<float>> uni = {{0.0f,0.0f},{0.1f,-0.1f}};
        std::vector<std::vector<double>> yuni = {{1.0,0.0},{1.0,0.0}};
        std::vector<int> stu = {10,10};
        const double t_uni = fit_temperature(question_kind::choice, uni, yuni, stu, calibration{});
        chk("fit.uni.bounded", t_uni >= 1.0/20.0 && t_uni <= 20.0 ? 1.0 : 0.0, 1.0);
        chk("fit.uni.improves", nll_over(t_uni, uni, yuni) <= nll_over(1.0, uni, yuni) + 1e-9 ? 1.0 : 0.0, 1.0);

        // nll: -log p at the labelled option.
        std::vector<double> pr = {0.25, 0.75};
        std::vector<double> lab = {0.0, 1.0};
        chk("nll", nll(pr, lab), -std::log(0.75));
    }

    std::printf("\nMAXDIFF=%.3e\n", maxdiff);
    return maxdiff > 1e-6 ? 1 : 0;
}
