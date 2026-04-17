clear; clc;
set(groot, 'defaultTextInterpreter',     'latex');
set(groot, 'defaultAxesTickLabelInterp', 'latex');
set(groot, 'defaultLegendInterpreter',   'latex');

%% -----------------------------------------------------------------------
%  1. Fixed parameters  (identical to simplified problem)
% -----------------------------------------------------------------------
p.H           = 1.0;
p.E           = 70e9;
p.rho         = 2700;
p.mp          = 2000;
p.ay          = 30;
p.FC          = p.mp * p.ay / 2;
p.sigma_allow = 150e6;
p.delta_allow = 2e-3;

%  Reference mass (order-of-magnitude estimate at midpoint design)
p.M_ref = p.rho * (2*1e-3*1.5 + 2*1e-3*p.H + ...
                   2*1e-3*sqrt(1.5^2+p.H^2) + 2*1e-3*p.H);

%% -----------------------------------------------------------------------
%  2. Bounds and variable scaling
%     x = [Lc;  Ac;   Av;   Ad]
% -----------------------------------------------------------------------
Lcmin  = 0.5;    Lcmax  = 3.0;
Amin   = 1e-5;   Amax   = 1e-2;

lb = [Lcmin; Amin;  Amin;  Amin];
ub = [Lcmax; Amax;  Amax;  Amax];

x_scale   = [Lcmax; Amax; Amax; Amax];
p.x_scale = x_scale;
lb_t      = lb ./ x_scale;
ub_t      = ub ./ x_scale;

%% -----------------------------------------------------------------------
%  3. Augmented Lagrangian parameters  (identical to simplified)
% -----------------------------------------------------------------------
mu        = 10.0;
beta      = 1.5;
mu_max    = 1e8;
eps_out   = 1e-7;
eps_in    = 1e-8;
max_out   = 300;
max_in    = 10000;
eta       = 0.25;
c1        = 1e-4;
alpha0    = 1.0;

%% -----------------------------------------------------------------------
%  4. Multiple starting points
%     Run from several starts to probe for local minima
% -----------------------------------------------------------------------
starts = {
    [Lcmax; Amax;  Amax;  Amax ] ./ x_scale,  'upper bounds'       ;
    [1.5;   5e-3;  5e-3;  5e-3 ] ./ x_scale,  'midpoint'           ;
    [0.5;   Amax;  Amax;  Amax ] ./ x_scale,  'Lc=Lcmin, A=Amax'   ;
    [3.0;   Amax;  Amax;  Amax ] ./ x_scale,  'Lc=Lcmax, A=Amax'   ;
    [1.0;   8e-3;  2e-3;  8e-3 ] ./ x_scale,  'stress-dominated'   ;
};

nStarts  = size(starts, 1);
results  = cell(nStarts, 1);

fprintf('=== Full Problem — Multi-start Augmented Lagrangian ===\n\n');

for s = 1:nStarts

    fprintf('--- Start %d/%d: %s ---\n', s, nStarts, starts{s,2});

    xt        = starts{s,1};
    lambda    = zeros(4,1);
    viol_prev = inf;
    mu_s      = mu;

    hst.M      = [];
    hst.viol   = [];
    hst.lambda = [];
    hst.mu     = [];
    hst.xt     = [];
    hst.kkt    = [];

    fprintf('%-5s  %-12s  %-12s  %-10s  %-12s\n', ...
        'Iter','M [kg]','Max viol','mu','KKT res');
    fprintf('%s\n', repmat('-',1,58));

    for k = 1:max_out

        %% Inner loop
        for inner = 1:max_in
            LA_cur = auglag(xt, lambda, mu_s, p);
            grad   = fd_grad(@(z) auglag(z, lambda, mu_s, p), xt);
            if norm(grad) < eps_in; break; end

            alpha = alpha0;
            for bt = 1:80
                xt_trial = clamp(xt - alpha*grad, lb_t, ub_t);
                if auglag(xt_trial, lambda, mu_s, p) <= ...
                        LA_cur + c1*(grad'*(xt_trial - xt))
                    break;
                end
                alpha = alpha * 0.5;
            end
            if alpha < 1e-14; break; end
            xt_new = clamp(xt - alpha*grad, lb_t, ub_t);
            if norm(xt_new - xt) < 1e-14; break; end
            xt = xt_new;
        end

        %% Evaluate
        x_phys         = xt .* x_scale;
        [M_cur, g_cur] = eval_norm(x_phys, p);
        max_viol        = max(0, max(g_cur));

        grad_M  = fd_grad(@(z) eval_mass(z.*x_scale,p)/p.M_ref, xt);
        Jg      = fd_grad_vec(@(z) get_constraints(z.*x_scale,p), xt);
        lam_eff = max(0, lambda + mu_s*g_cur);
        kkt_res = norm(grad_M + Jg'*lam_eff);

        hst.M(end+1)         = M_cur;
        hst.viol(end+1)      = max_viol;
        hst.lambda(:,end+1)  = lambda;
        hst.mu(end+1)        = mu_s;
        hst.xt(:,end+1)      = xt;
        hst.kkt(end+1)       = kkt_res;

        fprintf('%-5d  %-12.6f  %-12.4e  %-10.4e  %-12.4e\n', ...
            k, M_cur, max_viol, mu_s, kkt_res);

        if max_viol < eps_out && kkt_res < eps_out
            fprintf('Converged at iteration %d.\n\n', k);
            break;
        end

        lambda    = max(0, lambda + mu_s * g_cur);
        if max_viol > eta * viol_prev
            mu_s = min(beta * mu_s, mu_max);
        end
        viol_prev = max_viol;
    end

    %% Store result
    x_opt_s = xt .* x_scale;
    [M_s, g_phys_s, N_s, dC_s] = eval_full(x_opt_s, p);
    g_norm_s = [g_phys_s(1)/p.sigma_allow;
                g_phys_s(2)/p.sigma_allow;
                g_phys_s(3)/p.sigma_allow;
                g_phys_s(4)/p.delta_allow];

    results{s}.label   = starts{s,2};
    results{s}.x       = x_opt_s;
    results{s}.M       = M_s;
    results{s}.g_norm  = g_norm_s;
    results{s}.N       = N_s;
    results{s}.dC      = dC_s;
    results{s}.lambda  = max(0, lambda);
    results{s}.kkt     = kkt_res;
    results{s}.hist    = hst;
    results{s}.xt      = xt;
end

%% -----------------------------------------------------------------------
%  5. Identify best result
% -----------------------------------------------------------------------
M_vals   = cellfun(@(r) r.M, results);
viol_vals = cellfun(@(r) max(0, max(r.g_norm)), results);
feasible_mask = viol_vals < 1e-3;
M_vals(~feasible_mask) = inf;
[~, best_idx] = min(M_vals);
best = results{best_idx};

fprintf('=== Best result: start %d (%s) ===\n', ...
    best_idx, best.label);
fprintf('  Lc* = %.6f m\n',   best.x(1));
fprintf('  Ac* = %.6e m^2\n', best.x(2));
fprintf('  Av* = %.6e m^2\n', best.x(3));
fprintf('  Ad* = %.6e m^2\n', best.x(4));
fprintf('  M*  = %.6f kg\n',  best.M);
fprintf('  dC  = %.6e m  (limit %.6e m)\n', best.dC, p.delta_allow);
fprintf('  Nc  = %.2f N   Nv = %.2f N   Nd = %.2f N\n', ...
    best.N(1), best.N(2), best.N(3));

cnames = {'chord stress','vert stress','diag stress','deflection'};
fprintf('\n  Constraints (normalised):\n');
for i = 1:4
    fprintf('    g%d %-18s = %+.6f   lambda = %.4e   [%s]\n', ...
        i, cnames{i}, best.g_norm(i), best.lambda(i), ...
        active_str(best.g_norm(i)));
end

%% -----------------------------------------------------------------------
%  6. KKT verification for best result
% -----------------------------------------------------------------------
fprintf('\n=== KKT Conditions (best result) ===\n');
xt_best      = best.xt;
grad_M_kkt   = fd_grad(@(z) eval_mass(z.*x_scale,p)/p.M_ref, xt_best);
Jg_kkt       = fd_grad_vec(@(z) get_constraints(z.*x_scale,p), xt_best);
lam_star     = best.lambda;
stationarity = grad_M_kkt + Jg_kkt'*lam_star;
fprintf('  Stationarity residual = %.4e\n', norm(stationarity));
fprintf('  Complementary slackness:\n');
for i = 1:4
    fprintf('    lambda_%d * g~_%d = %+.4e\n', ...
        i, i, lam_star(i)*best.g_norm(i));
end

%% -----------------------------------------------------------------------
%  7. Comparison: simplified vs full
% -----------------------------------------------------------------------
M_simplified = 22.774034;   % from simplified problem
M_full       = best.M;
improvement  = (M_simplified - M_full) / M_simplified * 100;
fprintf('\n=== Comparison: simplified vs full problem ===\n');
fprintf('  M* simplified = %.4f kg  (Lc fixed = 1.5 m)\n', M_simplified);
fprintf('  M* full       = %.4f kg  (Lc free)\n', M_full);
fprintf('  Mass saving   = %.2f%%\n', improvement);

%% -----------------------------------------------------------------------
%  8. fmincon benchmark
% -----------------------------------------------------------------------
fprintf('\n=== fmincon benchmark ===\n');
opts_fmc = optimoptions('fmincon', ...
    'Algorithm',          'sqp', ...
    'Display',            'iter', ...
    'OptimalityTolerance', 1e-8, ...
    'ConstraintTolerance', 1e-8, ...
    'MaxIterations',       500);

obj_fmc = @(x) eval_mass(x, p);
con_fmc = @(x) deal(eval_constraints_phys(x, p), []);

[x_fmc, M_fmc, ~, out_fmc] = fmincon(obj_fmc, best.x, ...
    [], [], [], [], lb, ub, con_fmc, opts_fmc);

[~, g_fmc_phys, N_fmc, dC_fmc] = eval_full(x_fmc, p);
g_fmc_norm = [g_fmc_phys(1)/p.sigma_allow;
              g_fmc_phys(2)/p.sigma_allow;
              g_fmc_phys(3)/p.sigma_allow;
              g_fmc_phys(4)/p.delta_allow];

fprintf('\n  fmincon result:\n');
fprintf('  Lc* = %.6f m\n',   x_fmc(1));
fprintf('  Ac* = %.6e m^2\n', x_fmc(2));
fprintf('  Av* = %.6e m^2\n', x_fmc(3));
fprintf('  Ad* = %.6e m^2\n', x_fmc(4));
fprintf('  M*  = %.6f kg\n',  M_fmc);
fprintf('  Iterations: %d\n', out_fmc.iterations);
fprintf('  Agreement with AL: %.4e kg\n', abs(M_fmc - M_full));

%% -----------------------------------------------------------------------
%  9. Parametric study: effect of delta_allow on M*
% -----------------------------------------------------------------------
fprintf('\nRunning parametric study (M* vs delta_allow) ...\n');
da_vec    = linspace(0.5e-3, 5e-3, 20);
Mopt_da   = nan(size(da_vec));
Lc_da     = nan(size(da_vec));

for ii = 1:length(da_vec)
    p_par             = p;
    p_par.delta_allow = da_vec(ii);

    xt_par    = best.x ./ x_scale;   % warm start from best
    lam_par   = zeros(4,1);
    mu_par    = 10.0;
    vp_prev   = inf;

    for k = 1:max_out
        for inner = 1:max_in
            gp    = fd_grad(@(z) auglag_p(z,lam_par,mu_par,p_par), xt_par);
            if norm(gp) < eps_in; break; end
            LA_p  = auglag_p(xt_par, lam_par, mu_par, p_par);
            alpha = alpha0;
            for bt = 1:80
                xn = clamp(xt_par - alpha*gp, lb_t, ub_t);
                if auglag_p(xn,lam_par,mu_par,p_par) <= ...
                        LA_p + c1*(gp'*(xn-xt_par)); break; end
                alpha = alpha*0.5;
            end
            if alpha < 1e-14; break; end
            xt_new = clamp(xt_par - alpha*gp, lb_t, ub_t);
            if norm(xt_new-xt_par) < 1e-14; break; end
            xt_par = xt_new;
        end
        [~, g_c] = eval_norm(xt_par.*x_scale, p_par);
        vp_cur   = max(0, max(g_c));
        lam_par  = max(0, lam_par + mu_par*g_c);
        if vp_cur > eta*vp_prev
            mu_par = min(beta*mu_par, mu_max);
        end
        vp_prev = vp_cur;
        if vp_cur < eps_out; break; end
    end
    xp         = xt_par .* x_scale;
    Mopt_da(ii) = eval_mass(xp, p_par);
    Lc_da(ii)   = xp(1);
    fprintf('  delta_allow = %.2f mm  ->  M* = %.4f kg   Lc* = %.3f m\n', ...
        da_vec(ii)*1e3, Mopt_da(ii), Lc_da(ii));
end

%% -----------------------------------------------------------------------
%  10. Figures
% -----------------------------------------------------------------------

%% --- Figure 1: Multi-start comparison bar chart ----------------------
figure('Name','Multi-start Results','Position',[50 50 700 400]);

M_all    = cellfun(@(r) r.M,   results);
feas_all = cellfun(@(r) max(0,max(r.g_norm)), results) < 1e-3;
labels   = cellfun(@(r) r.label, results, 'UniformOutput', false);

bar_colors = repmat([0.4 0.6 0.9], nStarts, 1);
bar_colors(best_idx,:) = [0.2 0.7 0.3];
bar_colors(~feas_all,:) = repmat([0.8 0.3 0.3], sum(~feas_all), 1);

b = bar(M_all, 'FaceColor','flat');
b.CData = bar_colors;
set(gca, 'XTickLabel', labels, 'XTick', 1:nStarts, ...
    'TickLabelInterpreter','latex');
xtickangle(15);
yline(M_simplified, 'k--', 'LineWidth', 1.5);
text(nStarts*0.6, M_simplified*1.02, ...
    sprintf('Simplified: %.2f kg', M_simplified), ...
    'FontSize', 10);
ylabel('$M^*$ [kg]', 'FontSize', 12);
title('Multi-start results — full problem', 'FontSize', 12);
grid on; box on;
legend({'AL result','Best result','Infeasible'}, ...
    'Location','northeast','FontSize',9);

%% --- Figure 2: Convergence of best start -----------------------------
figure('Name','Full Problem Convergence','Position',[780 50 660 700]);
hst_best = best.hist;

subplot(4,1,1);
plot(hst_best.M,'b-o','MarkerSize',4,'LineWidth',1.5);
ylabel('$M$ [kg]','FontSize',11);
title('Objective — best start','FontSize',11);
grid on; xlim([1 length(hst_best.M)]);

subplot(4,1,2);
semilogy(max(hst_best.viol,1e-14),'r-s','MarkerSize',4,'LineWidth',1.5);
hold on;
yline(eps_out,'k--','LineWidth',1.2);
ylabel('Max violation','FontSize',11);
title('Constraint feasibility','FontSize',11);
grid on; xlim([1 length(hst_best.viol)]);

subplot(4,1,3);
semilogy(max(hst_best.kkt,1e-14),'m-^','MarkerSize',4,'LineWidth',1.5);
hold on;
yline(eps_out,'k--','LineWidth',1.2);
ylabel('KKT residual','FontSize',11);
title('Stationarity','FontSize',11);
grid on; xlim([1 length(hst_best.kkt)]);

subplot(4,1,4);
plot(hst_best.lambda','LineWidth',1.5);
ylabel('$\lambda_i$','FontSize',11);
xlabel('Outer iteration','FontSize',11);
title('Multiplier estimates','FontSize',11);
legend('$\lambda_1$','$\lambda_2$','$\lambda_3$','$\lambda_4$', ...
    'Location','best','FontSize',9);
grid on; xlim([1 size(hst_best.lambda,2)]);

%% --- Figure 3: AL vs fmincon result comparison -----------------------
figure('Name','AL vs fmincon','Position',[50 720 700 360]);

vars    = {'$L_c$ [m]','$A_c$ [m$^2$]','$A_v$ [m$^2$]','$A_d$ [m$^2$]'};
x_al    = best.x;
x_fminc = x_fmc;

x_al_norm    = x_al ./ x_fminc;
x_fminc_norm = x_fminc ./ x_fminc; % = 1

bar_data = [x_al_norm, x_fminc_norm];
b2 = bar(bar_data, 'grouped');
b2(1).FaceColor = [0.2 0.5 0.8];
b2(2).FaceColor = [0.8 0.4 0.2];
set(gca,'XTickLabel', vars, 'XTick', 1:4, ...
    'TickLabelInterpreter','latex','FontSize',10);
ylabel('Design variable value','FontSize',12);
title('Optimal design variables: AL vs \texttt{fmincon}','FontSize',12);
legend('Augmented Lagrangian','fmincon (SQP)','Location','northwest','FontSize',10);
grid on; box on;

%% --- Figure 4: Parametric study — M* vs delta_allow -----------------
figure('Name','Parametric Study','Position',[780 720 700 380]);

yyaxis left;
plot(da_vec*1e3, Mopt_da, 'b-o', 'MarkerSize',6,'LineWidth',1.8);
ylabel('$M^*$ [kg]','FontSize',12);

yyaxis right;
plot(da_vec*1e3, Lc_da, 'r-s', 'MarkerSize',6,'LineWidth',1.8);
ylabel('$L_c^*$ [m]','FontSize',12);

xline(p.delta_allow*1e3,'k--','LineWidth',1.3);
text(p.delta_allow*1e3+0.05, max(Mopt_da)*0.95, ...
    sprintf('$\\delta_{\\mathrm{allow}}=%.0f$ mm', p.delta_allow*1e3), ...
    'FontSize',10,'Interpreter','latex');
xlabel('$\delta_{\mathrm{allow}}$ [mm]','FontSize',12);
title('Effect of deflection limit on optimal mass and chord length', ...
    'FontSize',12);
legend('$M^*$','$L_c^*$','Location','northwest','FontSize',10);
grid on; box on;

%% =========================================================================
%  LOCAL FUNCTIONS
%% =========================================================================

function LA = auglag(xt, lambda, mu, p)
    x      = xt .* p.x_scale;
    M_norm = eval_mass(x, p) / p.M_ref;
    g      = get_constraints(x, p);
    pen    = 0;
    for i  = 1:length(g)
        pen = pen + (max(0, lambda(i) + mu*g(i))^2 - lambda(i)^2);
    end
    LA = M_norm + pen / (2*mu);
end

function LA = auglag_p(xt, lambda, mu, p)
    x      = xt .* p.x_scale;
    M_norm = eval_mass(x, p) / p.M_ref;
    g      = get_constraints(x, p);
    pen    = 0;
    for i  = 1:length(g)
        pen = pen + (max(0, lambda(i) + mu*g(i))^2 - lambda(i)^2);
    end
    LA = M_norm + pen / (2*mu);
end

function M = eval_mass(x, p)
    Lc  = x(1); Ac = x(2); Av = x(3); Ad = x(4);
    Ld  = sqrt(Lc^2 + p.H^2);
    M   = p.rho * (2*Ac*Lc + 2*Av*p.H + 2*Ad*Ld);
end

function [M, g_norm] = eval_norm(x, p)
    [M, g_phys, ~, ~] = eval_full(x, p);
    g_norm = [g_phys(1) / p.sigma_allow;
              g_phys(2) / p.sigma_allow;
              g_phys(3) / p.sigma_allow;
              g_phys(4) / p.delta_allow];
end

function g_norm = get_constraints(x, p)
    [~, g_norm] = eval_norm(x, p);
end

function g_phys = eval_constraints_phys(x, p)
%EVAL_CONSTRAINTS_PHYS  Physical constraint vector for fmincon.
    [~, g_phys, ~, ~] = eval_full(x, p);
end

function [M, g, N, dC] = eval_full(x, p)
    Lc = x(1); Ac = x(2); Av = x(3); Ad = x(4);
    Ld = sqrt(Lc^2 + p.H^2);

    % Nodes: 1=A(0,0)  2=B(Lc,0)  3=C(Lc,H)  4=D(0,H)
    coords = [0,  0;
              Lc, 0;
              Lc, p.H;
              0,  p.H];

    % Connectivity [nodeI, nodeJ, area]
    conn = [1, 2, Ac;   % bottom chord  A-B
            4, 3, Ac;   % top chord     D-C
            1, 4, Av;   % left vertical A-D
            2, 3, Av;   % right vertical B-C
            1, 3, Ad;   % diagonal A-C
            4, 2, Ad];  % diagonal D-B

    nDOF = 8;
    K    = zeros(nDOF);

    for e = 1:size(conn,1)
        ni = conn(e,1); nj = conn(e,2); Ae = conn(e,3);
        ci = coords(ni,:); cj = coords(nj,:);
        Le = norm(cj-ci);
        cv = (cj(1)-ci(1))/Le; sv = (cj(2)-ci(2))/Le;
        ke = (p.E*Ae/Le)*[ cv^2, cv*sv,-cv^2,-cv*sv;
                            cv*sv, sv^2,-cv*sv,-sv^2;
                           -cv^2,-cv*sv, cv^2, cv*sv;
                           -cv*sv,-sv^2, cv*sv, sv^2];
        dofs = [2*ni-1,2*ni,2*nj-1,2*nj];
        K(dofs,dofs) = K(dofs,dofs) + ke;
    end

    fixedDOF   = [1,2,7,8];
    freeDOF    = setdiff(1:nDOF, fixedDOF);
    f          = zeros(nDOF,1);
    f(6)       = -p.FC;
    u          = zeros(nDOF,1);
    u(freeDOF) = K(freeDOF,freeDOF) \ f(freeDOF);

    Nv = zeros(6,1);
    for e = 1:size(conn,1)
        ni = conn(e,1); nj = conn(e,2); Ae = conn(e,3);
        ci = coords(ni,:); cj = coords(nj,:);
        Le = norm(cj-ci);
        cv = (cj(1)-ci(1))/Le; sv = (cj(2)-ci(2))/Le;
        Nv(e) = (p.E*Ae/Le)*[-cv,-sv,cv,sv]*u([2*ni-1,2*ni,2*nj-1,2*nj]);
    end

    Nc    = max(abs(Nv(1:2)));
    Nvert = max(abs(Nv(3:4)));
    Nd    = max(abs(Nv(5:6)));
    N     = [Nc; Nvert; Nd];
    dC    = norm(u(5:6));

    M = p.rho * (2*Ac*Lc + 2*Av*p.H + 2*Ad*Ld);

    g = [Nc/Ac          - p.sigma_allow;
         Nvert/Av        - p.sigma_allow;
         Nd/Ad           - p.sigma_allow;
         dC              - p.delta_allow];
end

function grad = fd_grad(f, x)
    n    = length(x);
    grad = zeros(n,1);
    for j = 1:n
        h       = max(1e-5 * max(abs(x(j)), 1e-8), 1e-12);
        xp = x; xp(j) = x(j)+h;
        xm = x; xm(j) = x(j)-h;
        grad(j) = (f(xp)-f(xm))/(2*h);
    end
end

function J = fd_grad_vec(f, x)
    n  = length(x);
    f0 = f(x);
    m  = length(f0);
    J  = zeros(m,n);
    for j = 1:n
        h      = max(1e-5 * max(abs(x(j)), 1e-8), 1e-12);
        xp = x; xp(j) = x(j)+h;
        xm = x; xm(j) = x(j)-h;
        J(:,j) = (f(xp)-f(xm))/(2*h);
    end
    assert(size(J,1)==m && size(J,2)==n, ...
        'fd_grad_vec: unexpected size (%dx%d)', size(J,1), size(J,2));
end

function x = clamp(x, lb, ub)
    x = max(lb, min(ub, x));
end

function s = active_str(g_val)
    if abs(g_val) < 1e-2; s = 'ACTIVE'; else; s = 'inactive'; end
end