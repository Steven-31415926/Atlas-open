import ROOT
from ROOT import TCanvas, TH1F, TGraphErrors, TF1, TLegend, TLatex, TPad, TLine, gStyle
import vector
import math

# ======== 单事件处理函数 ========
def process_single_event(tree, entry_index, hist, is_mc=False):
    tree.GetEntry(entry_index)

    #双光子触发
    if not tree.trigP or tree.photon_pt.size() != 2:
        return False
    
    #读取与转换
    pt1 = tree.photon_pt[0] / 1000.0
    pt2 = tree.photon_pt[1] / 1000.0
    eta1 = tree.photon_eta[0]
    eta2 = tree.photon_eta[1]
    phi1 = tree.photon_phi[0]
    phi2 = tree.photon_phi[1]

    #排序
    if pt1 < pt2:
        pt1, pt2 = pt2, pt1
    
    #ptcut
    if pt1 < 35 or pt2 < 25:
        return False
    
    #计算不变质量
    m_gg = math.sqrt(2 * pt1 * pt2 * (math.cosh(eta1 - eta2) - math.cos(phi1 - phi2)))

    #排除错误事件
    if m_gg <= 0.0: 
        return False
    
    # 双光子质量比值cut
    if pt1 / m_gg < 0.35 or pt2 / m_gg < 0.25:
        return False
    
    #双光子质量cut
    if not (105 < m_gg < 160):
        return False
    
    #隔离度cut
    try:
        if (tree.photon_ptcone30[0] >= 0.065 or tree.photon_ptcone30[1] >= 0.065 or
            tree.photon_etcone20[0] >= 0.065 or tree.photon_etcone20[1] >= 0.065):
            return False
    except:
        pass
    
    #计算权重（仅MC）
    weight = 1.0
    if is_mc:
        try:
            weight = (tree.mcWeight * tree.scaleFactor_PHOTON * 
                     tree.scaleFactor_PhotonTRIGGER * tree.scaleFactor_PILEUP)
        except:
            weight = 1.0
    
    #直接填充直方图
    hist.Fill(m_gg, weight)
    return True

# ======== 主处理循环=======
def process_all_files(file_configs, data_hist, mc_hist):
    total_data = 0
    total_mc = 0
    
    for file_path, is_mc, label in file_configs:
        print(f"\n处理 {label}: {file_path}")
        
        root_file = ROOT.TFile(file_path, "READ")
        if not root_file or root_file.IsZombie():
            print(f"  错误：无法打开文件 {file_path}")
            continue
        
        tree = root_file.Get("mini")
        if not tree:
            print(f"  错误：无法获取树 'mini'")
            root_file.Close()
            continue
        
        n_entries = tree.GetEntries()
        passed = 0
        
        for i in range(n_entries):
            if process_single_event(tree, i, mc_hist if is_mc else data_hist, is_mc):
                passed += 1
        
        root_file.Close()
        print(f"  通过: {passed}/{n_entries}")
        
        if is_mc:
            total_mc += passed
        else:
            total_data += passed
    
    return total_data, total_mc

# ======== 运行主程序 ========
if __name__ == "__main__":
    file_configs = [
        ("data_A.GamGam.root", False, "Data A"),
        ("data_B.GamGam.root", False, "Data B"),
        ("data_C.GamGam.root", False, "Data C"),
        ("data_D.GamGam.root", False, "Data D"),
        ("mc_343981.ggH125_gamgam.GamGam.root", True, "MC")
    ]
    
    h_data = TH1F("h_data", "Data", 30, 105, 160)
    h_mc = TH1F("h_mc", "MC Signal", 30, 105, 160)
    h_mc.Sumw2()
    
    print("开始处理文件...")
    total_data, total_mc = process_all_files(file_configs, h_data, h_mc)
    
    print(f"\n{'='*50}")
    print(f"Data总通过事件数: {total_data}")
    print(f"MC总通过事件数: {total_mc}")
    
    # ======== 预拟合MC信号参数 ========
    print("\n预拟合MC信号获取高斯参数...")
    f_gauss = TF1("f_gauss", "gaus", 105, 160)
    f_gauss.SetParameters(100, 125.07, 2.51)
    h_mc.Fit(f_gauss, "RN")
    mc_mean = f_gauss.GetParameter(1)
    mc_sigma = f_gauss.GetParameter(2)
    print(f"MC高斯拟合: mean = {mc_mean:.2f} GeV, sigma = {mc_sigma:.2f} GeV")
    
    # ======== 联合拟合 ========
    f_total = TF1("f_total", "pol3(0) + gaus(4)", 105, 160)
    f_total.SetParameters(20455.4, -391.151, 2.57413, -0.00576837,
                          h_data.GetMaximum()*0.1, mc_mean, mc_sigma)
    f_total.SetParNames("a", "b", "c", "d", "Signal", "Mean", "Sigma")
    f_total.FixParameter(5, mc_mean)
    f_total.FixParameter(6, mc_sigma)
    
    h_data.Fit(f_total, "RN")
    signal_strength = f_total.GetParameter(4)
    signal_error = f_total.GetParError(4)
    
    # ======== 创建TGraphErrors用于绘图 ========
    g_data = TGraphErrors(h_data)
    g_data.SetMarkerStyle(20)
    g_data.SetMarkerSize(1)
    g_data.SetLineColor(ROOT.kBlack)
    
    # ======== ROOT 绘图 ========
    # 创建组合画布
    c = TCanvas("c", "双光子不变质量", 800, 800)

    # 创建上下两个Pad
    pad1 = TPad("pad1", "pad1", 0, 0.35, 1, 1)
    pad2 = TPad("pad2", "pad2", 0, 0, 1, 0.35)
    pad1.SetBottomMargin(0.005)
    pad1.SetLeftMargin(0.12)
    pad1.SetRightMargin(0.05)
    pad2.SetTopMargin(0.005)
    pad2.SetBottomMargin(0.35)
    pad2.SetLeftMargin(0.12)
    pad2.SetRightMargin(0.05)
    pad1.Draw()
    pad2.Draw()

    # 提前定义背景函数
    f_bkg = TF1("f_bkg", "pol3", 105, 160)
    for i in range(4):
        f_bkg.SetParameter(i, f_total.GetParameter(i))
    f_bkg.SetLineColor(ROOT.kBlue)
    f_bkg.SetLineStyle(2)

    # 定义信号线
    f_signal_only = TF1("f_signal_only", "gaus", 105, 160)
    f_signal_only.SetParameters(f_total.GetParameter(4), mc_mean, mc_sigma)
    f_signal_only.SetLineColor(ROOT.kBlack)
    f_signal_only.SetLineStyle(1)

    # 设置总拟合线样式
    f_total.SetLineColor(ROOT.kRed)
    f_total.SetLineStyle(1)

    # ========= 上方主图 =========
    pad1.cd()

    # 设置原始数据范围0-1000
    g_data.GetXaxis().SetRangeUser(105, 160)
    g_data.SetMinimum(0)
    g_data.SetMaximum(1000)

    # 绘制数据点
    g_data.Draw("AP")
    g_data.SetTitle("")

    # 绘制函数
    f_bkg.Draw("SAME")
    f_signal_only.Draw("SAME")
    f_total.Draw("SAME")

    # 标签
    latex = TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.04)
    latex.SetTextFont(72)
    latex.DrawLatex(0.25, 0.85, "ATLAS")
    latex.SetTextFont(42)
    latex.DrawLatex(0.40, 0.85, "Open Data")

    latex.SetTextSize(0.035)
    latex.DrawLatex(0.25, 0.79, "#sqrt{s} = 13 TeV, 10 fb^{-1}")
    latex.DrawLatex(0.25, 0.73, "H#rightarrow#gamma#gamma")

    # 图例
    leg = TLegend(0.65, 0.55, 0.88, 0.70)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(g_data, "Data", "PE")
    leg.AddEntry(f_total, "Signal + Background", "L")
    leg.AddEntry(f_bkg, "Background", "L")
    leg.AddEntry(f_signal_only, "Signal", "L")
    leg.Draw()

    g_data.GetYaxis().SetLabelSize(0)
    g_data.GetYaxis().SetNdivisions(6, 0, 0)
    g_data.GetYaxis().ChangeLabel(1, -1, -1, -1, -1, -1, "0.0")
    g_data.GetYaxis().ChangeLabel(2, -1, -1, -1, -1, -1, "0.2")
    g_data.GetYaxis().ChangeLabel(3, -1, -1, -1, -1, -1, "0.4")
    g_data.GetYaxis().ChangeLabel(4, -1, -1, -1, -1, -1, "0.6")
    g_data.GetYaxis().ChangeLabel(5, -1, -1, -1, -1, -1, "0.8")
    g_data.GetYaxis().ChangeLabel(6, -1, -1, -1, -1, -1, "1.0")
    g_data.GetYaxis().SetLabelSize(0.04)

    # 设置Y轴标题
    g_data.GetYaxis().SetTitle("Events / (1.8 GeV)")
    g_data.GetYaxis().SetTitleSize(0.05)
    g_data.GetYaxis().SetTitleOffset(1.0)

    latex.SetTextAlign(12)
    latex.SetTextAngle(0)
    latex.DrawLatex(0.17, 0.87, "#times10^{3}") 

    # ========= 下方子图 =========
    pad2.cd()

    # 创建Data-Bkg图
    g_data_bkg = TGraphErrors()
    g_data_bkg.SetMarkerStyle(20)
    g_data_bkg.SetMarkerSize(1)
    g_data_bkg.SetLineColor(ROOT.kBlack)

    point_index = 0
    for i in range(1, h_data.GetNbinsX() + 1):
        data_val = h_data.GetBinContent(i)
        data_err = h_data.GetBinError(i)
        bin_center = h_data.GetBinCenter(i)
        bkg_val = f_bkg.Eval(bin_center)
        
        if data_err > 0:
            g_data_bkg.SetPoint(point_index, bin_center, data_val - bkg_val)
            g_data_bkg.SetPointError(point_index, 0, data_err)
            point_index += 1

    g_data_bkg.GetXaxis().SetRangeUser(105, 160)
    g_data_bkg.GetYaxis().SetRangeUser(-40, 40)
    g_data_bkg.GetYaxis().SetNdivisions(505)

    # 绘制数据点
    g_data_bkg.Draw("AP")

    # 绘制0参考线
    line0 = TLine(105, 0, 160, 0)
    line0.SetLineStyle(2)
    line0.SetLineColor(ROOT.kGray+1)
    line0.Draw("SAME")

    # 绘制红色信号线
    f_signal_clone = f_signal_only.Clone("f_signal_clone")
    f_signal_clone.SetLineColor(ROOT.kRed)
    f_signal_clone.Draw("SAME")

    # 设置坐标轴
    g_data_bkg.GetXaxis().SetTitle("m_{#gamma#gamma} [GeV]")
    g_data_bkg.GetXaxis().SetTitleSize(0.12)
    g_data_bkg.GetXaxis().SetLabelSize(0.1)
    g_data_bkg.GetXaxis().SetTitleOffset(1.0)
    g_data_bkg.GetYaxis().SetTitle("Data - Bkg")
    g_data_bkg.GetYaxis().SetTitleSize(0.12)
    g_data_bkg.GetYaxis().SetLabelSize(0.08)
    g_data_bkg.GetYaxis().SetTitleOffset(0.4)

    input("绘图完成，按回车键退出")