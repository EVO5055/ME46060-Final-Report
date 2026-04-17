clear; clc;
set(groot, 'defaultTextInterpreter',     'latex');
set(groot, 'defaultAxesTickLabelInterp', 'latex');
set(groot, 'defaultLegendInterpreter',   'latex');

%% -----------------------------------------------------------------------
%  1. Fixed parameters
% -----------------------------------------------------------------------
p.H           = 1.0;       % [m]      Bay height
p.E           = 70e9;      % [Pa]     Young's modulus (aluminium)
p.rho         = 2700;      % [kg/m3]  Density
p.mp          = 2000;      % [kg]     Solar panel mass
p.ay          = 30;        % [m/s2]   Manoeuvre acceleration (~3g)
p.FC          = p.mp * p.ay / 2;   % [N]  Load at free node C
p.sigma_allow = 150e6;     % [Pa]     Allowable stress
p.delta_allow = 2e-3;      % [m]      Allowable deflection

%  Fixed geometry
p.Lc = 1.5;
p.Ld = sqrt(p.Lc^2 + p.H^2);

%  Reference mass for objective normalisation
p.M_ref = p.rho * (2*1e-3*p.Lc + 2*1e-3*p.H + 2*1e-3*p.Ld);

%% -----------------------------------------------------------------------
%  2. Bounds and variable scaling
% -----------------------------------------------------------------------
Amin       = 1e-5;
Amax       = 1e-2;
lb         = [Amin; Amin];
ub         = [Amax; Amax];

x_scale    = [Amax; Amax];   % xt = x / x_scale  =>  xt in [~0, 1]
p.x_scale  = x_scale;
lb_t       = lb ./ x_scale;
ub_t       = ub ./ x_scale;  % = [1; 1]

%% -----------------------------------------------------------------------
%  3. Augmented Lagrangian parameters
% -----------------------------------------------------------------------
mu        = 10.0;     % initial penalty
beta      = 1.5;      % penalty growth factor (conservative)
mu_max    = 1e8;      % hard cap — prevents blow-up
lambda    = zeros(4,1);
eps_out   = 1e-7;     % outer KKT tolerance
eps_in    = 1e-8;     % inner gradient tolerance
max_out   = 200;
max_in    = 10000;
eta       = 0.25;     % violation must drop by this factor to skip mu update
viol_prev = inf;

%  Armijo parameters
c1        = 1e-4;
alpha0    = 1.0;

%% -----------------------------------------------------------------------
%  4. Starting point — upper bounds (guaranteed feasible)
% -----------------------------------------------------------------------
xt = ub_t;   % scaled [1; 1] corresponds to physical [Amax; Amax]

x_phys = xt .* x_scale;
fprintf('Starting point: Acd = %.3e m^2,  Av = %.3e m^2\n', ...
    x_phys(1), x_phys(2));
[M0, g0] = eval_norm(x_phys, p);
fprintf('Starting mass:  M   = %.6f kg\n', M0);
fprintf('Starting constraints (normalised, <=0 feasible):\n');
fprintf('  g1=%+.4f  g2=%+.4f  g3=%+.4f  g4=%+.4f\n\n', ...
    g0(1), g0(2), g0(3), g0(4));

%% -----------------------------------------------------------------------
%  5. History storage
% -----------------------------------------------------------------------
hist.M      = [];
hist.viol   = [];
hist.lambda = [];
hist.mu     = [];
hist.xt     = [];
hist.kkt    = [];

%% -----------------------------------------------------------------------
%  6. Augmented Lagrangian outer loop
% -----------------------------------------------------------------------
fprintf('%-5s  %-12s  %-12s  %-10s  %-12s  %-12s\n', ...
    'Iter','M [kg]','Max viol','mu','||grad LA||','KKT res');
fprintf('%s\n', repmat('-',1,72));

for k = 1:max_out

    %% -- 6a. Inner loop: steepest descent + Armijo --------------------
    for inner = 1:max_in

        LA_cur = auglag(xt, lambda, mu, p);
        grad   = fd_grad(@(z) auglag(z, lambda, mu, p), xt);

        if norm(grad) < eps_in; break; end

        % Armijo backtracking line search
        alpha = alpha0;
        for bt = 1:80
            xt_trial = clamp(xt - alpha*grad, lb_t, ub_t);
            if auglag(xt_trial, lambda, mu, p) <= ...
                    LA_cur + c1*(grad'*(xt_trial - xt))
                break;
            end
            alpha = alpha * 0.5;
        end

        % Guard: step collapsed — no progress possible at this mu
        if alpha < 1e-14; break; end

        xt_new = clamp(xt - alpha*grad, lb_t, ub_t);
        if norm(xt_new - xt) < 1e-14; break; end
        xt = xt_new;
    end

    %% -- 6b. Evaluate at current point --------------------------------
    x_phys         = xt .* x_scale;
    [M_cur, g_cur] = eval_norm(x_phys, p);
    max_viol        = max(0, max(g_cur));
    grad_out        = fd_grad(@(z) auglag(z, lambda, mu, p), xt);

    %% -- 6c. KKT stationarity residual --------------------------------
    grad_M  = fd_grad(@(z) eval_mass(z.*x_scale, p)/p.M_ref, xt);
    Jg      = fd_grad_vec(@(z) get_constraints(z.*x_scale, p), xt);
    lam_eff = max(0, lambda + mu*g_cur);
    kkt_res = norm(grad_M + Jg'*lam_eff);

    %% -- 6d. Store history --------------------------------------------
    hist.M(end+1)         = M_cur;
    hist.viol(end+1)      = max_viol;
    hist.lambda(:,end+1)  = lambda;
    hist.mu(end+1)        = mu;
    hist.xt(:,end+1)      = xt;
    hist.kkt(end+1)       = kkt_res;

    fprintf('%-5d  %-12.6f  %-12.4e  %-10.4e  %-12.4e  %-12.4e\n', ...
        k, M_cur, max_viol, mu, norm(grad_out), kkt_res);

    %% -- 6e. Convergence check ----------------------------------------
    if max_viol < eps_out && kkt_res < eps_out
        fprintf('\nConverged at outer iteration %d.\n', k);
        break;
    end

    %% -- 6f. Conditional multiplier and penalty update ----------------
    lambda = max(0, lambda + mu * g_cur);

    % Only grow mu if violation has not decreased sufficiently
    if max_viol > eta * viol_prev
        mu = min(beta * mu, mu_max);
    end
    viol_prev = max_viol;
end

%% -----------------------------------------------------------------------
%  7. Report optimum
% -----------------------------------------------------------------------
x_opt = xt .* x_scale;
[M_opt, g_opt_phys, N_opt, dC_opt] = eval_full(x_opt, p);
g_opt_norm = [g_opt_phys(1)/p.sigma_allow;
              g_opt_phys(2)/p.sigma_allow;
              g_opt_phys(3)/p.sigma_allow;
              g_opt_phys(4)/p.delta_allow];

fprintf('\n=== Optimum ===\n');
fprintf('  Acd* = %.6e m^2\n',  x_opt(1));
fprintf('  Av*  = %.6e m^2\n',  x_opt(2));
fprintf('  M*   = %.6f kg\n',   M_opt);
fprintf('  Node C deflection = %.6e m  (limit %.6e m)\n', ...
    dC_opt, p.delta_allow);
fprintf('  Member forces:\n');
fprintf('    Nc = %.2f N    Nv = %.2f N    Nd = %.2f N\n', ...
    N_opt(1), N_opt(2), N_opt(3));

cnames   = {'chord/diag stress','vert stress','diag stress','deflection'};
lam_star = max(0, lambda);
fprintf('\n  Constraints (normalised, 0 = active):\n');
for i = 1:4
    fprintf('    g%d %-22s = %+.6f   lambda = %.4e   [%s]\n', ...
        i, cnames{i}, g_opt_norm(i), lam_star(i), ...
        active_str(g_opt_norm(i)));
end

%% -----------------------------------------------------------------------
%  8. KKT verification
% -----------------------------------------------------------------------
fprintf('\n=== KKT Conditions ===\n');
grad_M_kkt   = fd_grad(@(z) eval_mass(z.*x_scale,p)/p.M_ref, xt);
Jg_kkt       = fd_grad_vec(@(z) get_constraints(z.*x_scale,p), xt);
stationarity = grad_M_kkt + Jg_kkt'*lam_star;
fprintf('  Stationarity residual = %.4e\n', norm(stationarity));
fprintf('  Dual feasibility lambda_i >= 0: ');
fprintf('%.3e  ', lam_star); fprintf('\n');
fprintf('  Complementary slackness (normalised):\n');
for i = 1:4
    fprintf('    lambda_%d * g~_%d = %+.4e\n', ...
        i, i, lam_star(i)*g_opt_norm(i));
end

%% -----------------------------------------------------------------------
%  9. Parametric study: M* vs Lc
% -----------------------------------------------------------------------
fprintf('\nRunning parametric study (M* vs Lc) ...\n');
Lc_vec   = linspace(0.5, 3.0, 25);
Mopt_vec = nan(size(Lc_vec));

for ii = 1:length(Lc_vec)
    p_par         = p;
    p_par.Lc      = Lc_vec(ii);
    p_par.Ld      = sqrt(p_par.Lc^2 + p_par.H^2);
    p_par.M_ref   = p.rho*(2*1e-3*p_par.Lc + ...
                           2*1e-3*p_par.H   + ...
                           2*1e-3*p_par.Ld);
    p_par.x_scale = x_scale;

    xt_par    = ub_t;
    lam_par   = zeros(4,1);
    mu_par    = 10.0;
    vp_prev   = inf;

    for k = 1:max_out
        for inner = 1:max_in
            gp = fd_grad(@(z) auglag_p(z,lam_par,mu_par,p_par), xt_par);
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
            if norm(xt_new - xt_par) < 1e-14; break; end
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
    Mopt_vec(ii) = eval_mass(xt_par.*x_scale, p_par);
    fprintf('  Lc = %.2f m  ->  M* = %.4f kg\n', Lc_vec(ii), Mopt_vec(ii));
end

%% -----------------------------------------------------------------------
%  10. Figures
% -----------------------------------------------------------------------

%% --- Figure 1: Feasible region and objective contours -----------------
figure('Name','Feasible Region','Position',[50 50 760 640]);

Npts = 280;
% Zoom axis around expected optimum region
A_lo  = Amin;
A_hi  = 2e-3;
Avec  = linspace(A_lo, A_hi, Npts);
[ACD, AV] = meshgrid(Avec, Avec);

G1 = nan(Npts); G2 = nan(Npts);
G3 = nan(Npts); G4 = nan(Npts);
MS = nan(Npts);

for ii = 1:Npts
    for jj = 1:Npts
        xi = [ACD(ii,jj); AV(ii,jj)];
        [Mi, gi, ~, ~] = eval_full(xi, p);
        G1(ii,jj) = gi(1); G2(ii,jj) = gi(2);
        G3(ii,jj) = gi(3); G4(ii,jj) = gi(4);
        MS(ii,jj) = Mi;
    end
end

feasible    = (G1<=0) & (G2<=0) & (G3<=0) & (G4<=0);
infeas_mask = double(~feasible);
infeas_mask(feasible) = NaN;

contourf(ACD, AV, MS, 25, 'LineWidth', 0.3); hold on;
colormap(parula(256));
cb = colorbar;
cb.Label.String    = 'Mass $M$ [kg]';
cb.Label.Interpreter = 'latex';
cb.Label.FontSize  = 11;

[~,hg1] = contour(ACD,AV,G1,[0 0],'r-', 'LineWidth',2.2);
[~,hg2] = contour(ACD,AV,G2,[0 0],'b-', 'LineWidth',2.2);
[~,hg3] = contour(ACD,AV,G3,[0 0],'m-', 'LineWidth',2.2);
[~,hg4] = contour(ACD,AV,G4,[0 0],'k--','LineWidth',2.2);

% Optimum
plot(x_opt(1), x_opt(2), 'kp','MarkerSize',18, ...
    'MarkerFaceColor','yellow','LineWidth',1.5);
text(x_opt(1)*1.04, x_opt(2)*1.04, 'Optimum','FontSize',10);

% Iteration path
x_hist_phys = hist.xt .* x_scale;
if size(x_hist_phys,2) > 1
    plot(x_hist_phys(1,:), x_hist_phys(2,:), 'w.-', ...
        'MarkerSize',10,'LineWidth',1.4);
end

xlabel('$A_{cd}$ [m$^2$]','FontSize',12);
ylabel('$A_v$ [m$^2$]','FontSize',12);
title('Feasible region and objective contours — simplified problem', ...
    'FontSize',12);
legend([hg1,hg2,hg3,hg4], ...
    '$g_1$ (chord stress)', '$g_2$ (vert.\ stress)', ...
    '$g_3$ (diag.\ stress)', '$g_4$ (deflection)', ...
    'Location','northeast','FontSize',10);
grid on; box on;

%% --- Figure 2: Convergence history ------------------------------------
figure('Name','AL Convergence','Position',[820 50 660 740]);

subplot(4,1,1);
plot(hist.M,'b-o','MarkerSize',4,'LineWidth',1.5);
ylabel('$M$ [kg]','FontSize',11);
title('Objective','FontSize',11);
grid on; xlim([1 length(hist.M)]);

subplot(4,1,2);
semilogy(max(hist.viol,1e-14),'r-s','MarkerSize',4,'LineWidth',1.5);
hold on;
yline(eps_out,'k--','LineWidth',1.2);
ylabel('Max violation','FontSize',11);
title('Constraint feasibility (normalised)','FontSize',11);
grid on; xlim([1 length(hist.viol)]);

subplot(4,1,3);
semilogy(max(hist.kkt,1e-14),'m-^','MarkerSize',4,'LineWidth',1.5);
hold on;
yline(eps_out,'k--','LineWidth',1.2);
ylabel('KKT residual','FontSize',11);
title('Stationarity','FontSize',11);
grid on; xlim([1 length(hist.kkt)]);

subplot(4,1,4);
plot(hist.lambda','LineWidth',1.5);
ylabel('$\lambda_i$','FontSize',11);
xlabel('Outer iteration','FontSize',11);
title('Multiplier estimates','FontSize',11);
legend('$\lambda_1$','$\lambda_2$','$\lambda_3$','$\lambda_4$', ...
    'Location','best','FontSize',9);
grid on; xlim([1 size(hist.lambda,2)]);

%% --- Figure 3: Parametric study M* vs Lc ------------------------------
figure('Name','Parametric Study','Position',[50 720 700 380]);

plot(Lc_vec, Mopt_vec,'b-o','MarkerSize',6,'LineWidth',1.8); hold on;
xline(p.Lc,'k--','LineWidth',1.3);
[~,idx] = min(Mopt_vec);
plot(Lc_vec(idx), Mopt_vec(idx),'rp','MarkerSize',16, ...
    'MarkerFaceColor','red','LineWidth',1.5);
text(p.Lc+0.06, max(Mopt_vec)*0.98, ...
    sprintf('$L_{c,0}=%.1f$ m', p.Lc),'FontSize',10);
text(Lc_vec(idx)+0.06, Mopt_vec(idx)*1.02, ...
    sprintf('$L_c^*=%.2f$ m', Lc_vec(idx)),'FontSize',10,'Color','red');
xlabel('$L_c$ [m]','FontSize',12);
ylabel('$M^*$ [kg]','FontSize',12);
title('Optimal mass vs.\ chord length (simplified problem)','FontSize',12);
grid on; box on;

%% =========================================================================
%  LOCAL FUNCTIONS
%% =========================================================================

function LA = auglag(xt, lambda, mu, p)
%AUGLAG  Augmented Lagrangian evaluated in scaled variable space.
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
%AUGLAG_P  Augmented Lagrangian for parametric study (variable p).
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
%EVAL_MASS  Total truss mass [kg].
    Acd = x(1); Av = x(2);
    M   = p.rho * (2*Acd*(p.Lc + p.Ld) + 2*Av*p.H);
end

function [M, g_norm] = eval_norm(x, p)
%EVAL_NORM  Mass and normalised constraints (dimensionless, <=0 feasible).
    [M, g_phys, ~, ~] = eval_full(x, p);
    g_norm = [g_phys(1) / p.sigma_allow;
              g_phys(2) / p.sigma_allow;
              g_phys(3) / p.sigma_allow;
              g_phys(4) / p.delta_allow];
end

function g_norm = get_constraints(x, p)
%GET_CONSTRAINTS  Normalised constraint vector only (4x1).
%  Isolates second output of eval_norm for use in fd_grad_vec.
    [~, g_norm] = eval_norm(x, p);
end

function [M, g, N, dC] = eval_full(x, p)
%EVAL_FULL  Full stiffness solution.
%  Returns mass, physical constraints, member forces, tip deflection.
    Acd = x(1); Av = x(2);

    % Nodes: 1=A(0,0)  2=B(Lc,0)  3=C(Lc,H)  4=D(0,H)
    coords = [0,    0;
              p.Lc, 0;
              p.Lc, p.H;
              0,    p.H];

    % Member connectivity [nodeI, nodeJ, area]
    conn = [1, 2, Acd;   % bottom chord  A-B
            4, 3, Acd;   % top chord     D-C
            1, 4, Av;    % left vertical A-D
            2, 3, Av;    % right vertical B-C
            1, 3, Acd;   % diagonal A-C
            4, 2, Acd];  % diagonal D-B

    nDOF = 8;
    K    = zeros(nDOF);

    for e = 1:size(conn,1)
        ni = conn(e,1); nj = conn(e,2); Ae = conn(e,3);
        ci = coords(ni,:); cj = coords(nj,:);
        Le = norm(cj - ci);
        cv = (cj(1)-ci(1))/Le;
        sv = (cj(2)-ci(2))/Le;
        ke = (p.E*Ae/Le) * ...
             [ cv^2,  cv*sv, -cv^2, -cv*sv;
               cv*sv, sv^2,  -cv*sv,-sv^2;
              -cv^2, -cv*sv,  cv^2,  cv*sv;
              -cv*sv,-sv^2,   cv*sv, sv^2];
        dofs = [2*ni-1, 2*ni, 2*nj-1, 2*nj];
        K(dofs,dofs) = K(dofs,dofs) + ke;
    end

    % Boundary conditions: pin at A (DOFs 1,2) and D (DOFs 7,8)
    fixedDOF   = [1, 2, 7, 8];
    freeDOF    = setdiff(1:nDOF, fixedDOF);
    f          = zeros(nDOF, 1);
    f(6)       = -p.FC;   % downward load at C (node 3, DOF 6)
    u          = zeros(nDOF, 1);
    u(freeDOF) = K(freeDOF,freeDOF) \ f(freeDOF);

    % Member axial forces
    Nv = zeros(6,1);
    for e = 1:size(conn,1)
        ni = conn(e,1); nj = conn(e,2); Ae = conn(e,3);
        ci = coords(ni,:); cj = coords(nj,:);
        Le = norm(cj - ci);
        cv = (cj(1)-ci(1))/Le;
        sv = (cj(2)-ci(2))/Le;
        Nv(e) = (p.E*Ae/Le)*[-cv,-sv,cv,sv] * ...
                u([2*ni-1,2*ni,2*nj-1,2*nj]);
    end

    Nc    = max(abs(Nv(1:2)));   % governing chord force
    Nvert = max(abs(Nv(3:4)));   % governing vertical force
    Nd    = max(abs(Nv(5:6)));   % governing diagonal force
    N     = [Nc; Nvert; Nd];
    dC    = norm(u(5:6));        % tip deflection at C

    M = p.rho * (2*Acd*(p.Lc + p.Ld) + 2*Av*p.H);

    % Physical constraints (<=0 feasible)
    g = [max(Nc,Nd)/Acd - p.sigma_allow;   % combined chord/diag stress
         Nvert/Av        - p.sigma_allow;   % vertical stress
         Nd/Acd          - p.sigma_allow;   % diagonal stress
         dC              - p.delta_allow];  % tip deflection
end

function grad = fd_grad(f, x)
%FD_GRAD  Central finite difference gradient of scalar-valued function.
    n    = length(x);
    grad = zeros(n,1);
    for j = 1:n
        h       = max(1e-5 * max(abs(x(j)), 1e-8), 1e-12);
        xp      = x; xp(j) = x(j) + h;
        xm      = x; xm(j) = x(j) - h;
        grad(j) = (f(xp) - f(xm)) / (2*h);
    end
end

function J = fd_grad_vec(f, x)
%FD_GRAD_VEC  Jacobian of vector-valued function f: R^n -> R^m.
%  Returns J of size (m x n).
    n  = length(x);
    f0 = f(x);
    m  = length(f0);
    J  = zeros(m, n);
    for j = 1:n
        h      = max(1e-5 * max(abs(x(j)), 1e-8), 1e-12);
        xp     = x; xp(j) = x(j) + h;
        xm     = x; xm(j) = x(j) - h;
        J(:,j) = (f(xp) - f(xm)) / (2*h);
    end
    assert(size(J,1)==m && size(J,2)==n, ...
        'fd_grad_vec: unexpected size (%dx%d)', size(J,1), size(J,2));
end

function x = clamp(x, lb, ub)
%CLAMP  Project x onto box [lb, ub].
    x = max(lb, min(ub, x));
end

function s = active_str(g_val)
%ACTIVE_STR  Constraint activity label — expects normalised g (dimensionless).
    if abs(g_val) < 1e-2; s = 'ACTIVE'; else; s = 'inactive'; end
end