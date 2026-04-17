clear; clc;

%% -----------------------------------------------------------------------
%  1. Fixed parameters
% -----------------------------------------------------------------------
p.H             = 1.0;       % [m]      Fixed bay height
p.E             = 70e9;      % [Pa]     Young's modulus (aluminium 6061)
p.rho           = 2700;      % [kg/m3]  Density
p.mp            = 50;        % [kg]     Solar panel mass
p.ay            = 10;        % [m/s2]   Manoeuvre acceleration
p.sigma_allow   = 150e6;     % [Pa]     Allowable stress
p.delta_allow   = 5e-3;      % [m]      Allowable deflection

%  Applied load (only F_C enters the structure; F_D taken by support)
p.FC = p.mp * p.ay / 2;      % [N]

%% -----------------------------------------------------------------------
%  2. Baseline design vector  x = [Lc, Ac, Av, Ad]
% -----------------------------------------------------------------------
Lc0 = 1.5;       % [m]
Ac0 = 5e-4;      % [m2]
Av0 = 5e-4;      % [m2]
Ad0 = 5e-4;      % [m2]
x0  = [Lc0; Ac0; Av0; Ad0];

varNames = {'L_c', 'A_c', 'A_v', 'A_d'};

%% -----------------------------------------------------------------------
%  3. Evaluate at baseline
% -----------------------------------------------------------------------
[M0, g0, N0, dC0] = evaluate(x0, p);

fprintf('=== Baseline design ===\n');
fprintf('  x0 = [Lc=%.3f m, Ac=%.2e m2, Av=%.2e m2, Ad=%.2e m2]\n', ...
    x0(1), x0(2), x0(3), x0(4));
fprintf('  Mass M  = %.4f kg\n',  M0);
fprintf('  g1 (chord  stress) = %+.4e  (%.1f%% of limit)\n', ...
    g0(1), (g0(1)+p.sigma_allow)/p.sigma_allow*100);
fprintf('  g2 (vert   stress) = %+.4e  (%.1f%% of limit)\n', ...
    g0(2), (g0(2)+p.sigma_allow)/p.sigma_allow*100);
fprintf('  g3 (diag   stress) = %+.4e  (%.1f%% of limit)\n', ...
    g0(3), (g0(3)+p.sigma_allow)/p.sigma_allow*100);
fprintf('  g4 (deflection)    = %+.4e  (%.1f%% of limit)\n', ...
    g0(4), (g0(4)+p.delta_allow)/p.delta_allow*100);
fprintf('  Member forces [N]: Nc=%.2f  Nv=%.2f  Nd=%.2f\n\n', ...
    N0(1), N0(2), N0(3));

%% -----------------------------------------------------------------------
%  4. Central finite difference sensitivities
% -----------------------------------------------------------------------
h_rel  = 1e-4;          % Relative step size
nVar   = length(x0);
nCon   = length(g0);

dM_dx  = zeros(nVar, 1);
dg_dx  = zeros(nCon, nVar);

for j = 1:nVar
    h       = h_rel * x0(j);
    xp      = x0;  xp(j) = x0(j) + h;
    xm      = x0;  xm(j) = x0(j) - h;

    [Mp, gp, ~, ~] = evaluate(xp, p);
    [Mm, gm, ~, ~] = evaluate(xm, p);

    dM_dx(j)   = (Mp - Mm) / (2*h);
    dg_dx(:,j) = (gp - gm) / (2*h);
end

%% -----------------------------------------------------------------------
%  5. Normalised objective sensitivities  S_i = (x_i/M) * dM/dx_i
% -----------------------------------------------------------------------
S_norm = (x0 ./ M0) .* dM_dx;

%% -----------------------------------------------------------------------
%  6. Analytical objective sensitivities (verification)
% -----------------------------------------------------------------------
Lc = x0(1); Ac = x0(2); Av = x0(3); Ad = x0(4);
Ld = sqrt(Lc^2 + p.H^2);

dM_dLc_anal = p.rho * (2*Ac + 2*Ad * Lc/Ld);
dM_dAc_anal = p.rho * 2 * Lc;
dM_dAv_anal = p.rho * 2 * p.H;
dM_dAd_anal = p.rho * 2 * Ld;
dM_anal = [dM_dLc_anal; dM_dAc_anal; dM_dAv_anal; dM_dAd_anal];

%% -----------------------------------------------------------------------
%  7. Print sensitivity table
% -----------------------------------------------------------------------
fprintf('=== Objective sensitivity (central FD vs analytical) ===\n');
fprintf('%-6s  %12s  %12s  %12s\n', ...
    'Var', 'FD', 'Analytical', 'Normalised S');
for j = 1:nVar
    fprintf('%-6s  %12.4e  %12.4e  %12.4f\n', ...
        varNames{j}, dM_dx(j), dM_anal(j), S_norm(j));
end

fprintf('\n=== Constraint sensitivities (central FD) ===\n');
conNames = {'g1 chord','g2 vert','g3 diag','g4 defl'};
header = sprintf('%-12s', 'Constraint');
for j = 1:nVar
    header = [header, sprintf('%14s', varNames{j})]; %#ok<AGROW>
end
fprintf('%s\n', header);
for i = 1:nCon
    row = sprintf('%-12s', conNames{i});
    for j = 1:nVar
        row = [row, sprintf('%14.4e', dg_dx(i,j))]; %#ok<AGROW>
    end
    fprintf('%s\n', row);
end

%% -----------------------------------------------------------------------
%  8. Sensitivity bar charts
% -----------------------------------------------------------------------
figure('Name','Sensitivity Analysis','NumberTitle','off','Position',[100 100 900 600]);

% --- Normalised objective sensitivity ---
subplot(2,2,1);
bar(S_norm, 'FaceColor', [0.2 0.5 0.8]);
set(gca, 'XTickLabel', varNames, 'XTick', 1:nVar);
ylabel('Normalised sensitivity $\bar{S}_i$', 'Interpreter', 'latex');
title('Objective M — normalised sensitivity');
grid on; yline(0,'k--');

% --- Objective raw sensitivities ---
subplot(2,2,2);
bar(dM_dx, 'FaceColor', [0.2 0.7 0.4]);
set(gca, 'XTickLabel', varNames, 'XTick', 1:nVar);
ylabel('\partial M / \partial x_i');
title('Objective M — raw sensitivity');
grid on;

% --- Constraint sensitivities heatmap ---
subplot(2,1,2);
% Normalise each constraint row by its value for visual comparison
dg_norm = dg_dx ./ abs(g0 + 1e-12);   % avoid /0 if constraint = 0
imagesc(dg_norm);
colormap(redblue(256));
colorbar;
set(gca, 'XTickLabel', varNames,  'XTick', 1:nVar, ...
         'YTickLabel', conNames,   'YTick', 1:nCon);
title('Normalised constraint sensitivities  (\partial g_i/\partial x_j) / |g_i|');
xlabel('Design variable'); ylabel('Constraint');

%% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================

function [M, g, Nmembers, dC] = evaluate(x, p)
%EVALUATE  Compute mass, constraints, member forces and tip deflection.
%
%  Inputs:
%    x = [Lc; Ac; Av; Ad]  design vector
%    p                      fixed parameter struct
%
%  Outputs:
%    M         total truss mass [kg]
%    g         constraint vector [g1;g2;g3;g4] (<=0 feasible)
%    Nmembers  member forces [Nc; Nv; Nd] [N]
%    dC        resultant displacement of node C [m]

    Lc = x(1);  Ac = x(2);  Av = x(3);  Ad = x(4);
    Ld = sqrt(Lc^2 + p.H^2);

    %% --- Node coordinates ---
    % Node 1: A  (0,   0  )  pin support
    % Node 2: B  (Lc,  0  )  free
    % Node 3: C  (Lc,  H  )  free, loaded
    % Node 4: D  (0,   H  )  pin support
    coords = [0,   0;
              Lc,  0;
              Lc,  p.H;
              0,   p.H];

    %% --- Member connectivity [nodeI, nodeJ, area] ---
    %  Member 1: bottom chord  A-B
    %  Member 2: top chord     D-C
    %  Member 3: left vertical A-D
    %  Member 4: right vertical B-C
    %  Member 5: diagonal A-C
    %  Member 6: diagonal D-B
    conn = [1, 2, Ac;   % bottom chord
            4, 3, Ac;   % top chord
            1, 4, Av;   % left vertical
            2, 3, Av;   % right vertical
            1, 3, Ad;   % diagonal A-C
            4, 2, Ad];  % diagonal D-B

    nNodes  = 4;
    nMem    = size(conn, 1);
    nDOF    = 2 * nNodes;   % 8 total DOFs

    %% --- Assemble global stiffness matrix ---
    K = zeros(nDOF, nDOF);

    for e = 1:nMem
        ni = conn(e,1);  nj = conn(e,2);  Ae = conn(e,3);

        xi = coords(ni,:);  xj = coords(nj,:);
        Le = norm(xj - xi);
        c  = (xj(1)-xi(1)) / Le;   % cos(theta)
        s  = (xj(2)-xi(2)) / Le;   % sin(theta)

        ke = (p.E * Ae / Le) * ...
             [ c^2,  c*s, -c^2, -c*s;
               c*s,  s^2, -c*s, -s^2;
              -c^2, -c*s,  c^2,  c*s;
              -c*s, -s^2,  c*s,  s^2];

        dofs = [2*ni-1, 2*ni, 2*nj-1, 2*nj];
        K(dofs, dofs) = K(dofs, dofs) + ke;
    end

    %% --- Apply boundary conditions ---
    % Pin at A (node 1): DOFs 1,2 fixed
    % Pin at D (node 4): DOFs 7,8 fixed
    fixedDOF = [1, 2, 7, 8];
    freeDOF  = setdiff(1:nDOF, fixedDOF);

    %% --- External load vector ---
    % F_C downward at node C (node 3, DOF 5=x, 6=y)
    f = zeros(nDOF, 1);
    f(6) = -p.FC;   % downward (negative y)

    %% --- Solve for displacements ---
    Kff = K(freeDOF, freeDOF);
    ff  = f(freeDOF);
    uf  = Kff \ ff;

    u = zeros(nDOF, 1);
    u(freeDOF) = uf;

    %% --- Recover member forces ---
    N = zeros(nMem, 1);
    for e = 1:nMem
        ni = conn(e,1);  nj = conn(e,2);  Ae = conn(e,3);

        xi = coords(ni,:);  xj = coords(nj,:);
        Le = norm(xj - xi);
        c  = (xj(1)-xi(1)) / Le;
        s  = (xj(2)-xi(2)) / Le;

        dofsI = [2*ni-1, 2*ni];
        dofsJ = [2*nj-1, 2*nj];
        dU    = [u(dofsJ); u(dofsI)];  % elongation direction

        N(e) = (p.E * Ae / Le) * [-c, -s, c, s] * [u(dofsI); u(dofsJ)];
    end

    % Group by member type (max stress per type)
    Nc = max(abs(N(1:2)));   % chord members  1,2
    Nv = max(abs(N(3:4)));   % vertical members 3,4
    Nd = max(abs(N(5:6)));   % diagonal members 5,6
    Nmembers = [Nc; Nv; Nd];

    %% --- Tip deflection at node C (node 3) ---
    uC  = u(5:6);            % x and y displacement of C
    dC  = norm(uC);          % resultant

    %% --- Objective ---
    M = p.rho * (2*Ac*Lc + 2*Av*p.H + 2*Ad*Ld);

    %% --- Constraints (<=0 feasible) ---
    g = zeros(4, 1);
    g(1) = Nc / Ac - p.sigma_allow;
    g(2) = Nv / Av - p.sigma_allow;
    g(3) = Nd / Ad - p.sigma_allow;
    g(4) = dC      - p.delta_allow;
end

%% -----------------------------------------------------------------------
function cmap = redblue(n)
%REDBLUE  Red-white-blue diverging colormap for sensitivity heatmap.
    if nargin < 1; n = 256; end
    half = floor(n/2);
    r1 = linspace(1,1,half)';  g1 = linspace(0,1,half)';  b1 = linspace(0,1,half)';
    r2 = linspace(1,0,n-half)'; g2 = linspace(1,0,n-half)'; b2 = linspace(1,1,n-half)';
    cmap = [[r1;r2], [g1;g2], [b1;b2]];
end