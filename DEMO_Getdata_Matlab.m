%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Getdata Demo notebook (MATLAB)
% 
% supported datasets :
% 
%         - isotropic1024coarse  :  isotropic 1024-cube (coarse).
%         - isotropic1024fine       :  isotropic 1024-cube (fine).
%         - isotropic4096            :  isotropic 4096-cube.
%         - isotropic8192            :  isotropic 8192-cube.
%         - sabl2048low              :  stable atmospheric boundary layer 2048-cube, low-rate timestep.
%         - sabl2048high             :  stable atmospheric boundary layer 2048-cube, high-rate timestep.
%         - rotstrat4096               :  rotating stratified 4096-cube.
%         - mhd1024                   :  magneto-hydrodynamic isotropic 1024-cube.
%         - mixing                       :   homogeneous buoyancy driven 1024-cube.
%         - channel                     :  channel flow.
%         - channel5200              :  channel flow (reynolds number 5200).
%         - transition_bl               :  transitional boundary layer.
% 
% functions :
% 
%         - getData  :  retrieve (interpolate and/or differentiate) field data on a set of specified spatial points for the specified variable.        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% instantiate dataset 
%
% purpose :
%        - instantiate the dataset and cache the metadata.
%
% parameters :
% 
%        - auth_token    :  turbulence user authorization token.
%        - dataset_title  :  name of the turbulence dataset.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
clear all;
close all;

% ---- Enter user JHTDB token ----
authkey = 'edu.jhu.pha.turbulence.testing-201406';  
% the above is a default testing token that works for queries up to 4096 points
% for larger queries, please request token at Please send an e-mail to 
% turbulence@lists.johnshopkins.edu including your name, email address, 
% and institutional affiliation and department, together with a short 
% description of your intended use of the database.
%
% ---- select dataset ----
dataset =  'channel';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% getData 
%
% purpose :
%        - retrieve (interpolate and/or differentiate) a group of sparse data points.
%
% steps :
% 
%          - step 1  :  identify the database files to be read.
%          - step 2  :  read the database files and store the interpolated points in an array.
% 
% parameters :
% 
%          - dataset  :  the instantiated dataset.
%          - points  :  array of points in the domain [0, 2pi).
%          - variable  :  type of data (velocity, pressure, energy, temperature, force, magneticfield, vectorpotential, density, position).
%          - time  :  time point (snapshot number for datasets without a full time evolution).
%          - time_end  :  ending time point for 'position' variable and time series queries.
%          - delta_t  :  time step for 'position' variable and time series queries.
%          - temporal_method  :  temporal interpolation methods.
%                 - none  :  No temporal interpolation (the value at the closest stored time will be returned).
%                 - pchip  :  Piecewise Cubic Hermite Interpolation Polynomial method is used, in which the value from the two nearest time points
%                    is interpolated at time t using Cubic Hermite Interpolation Polynomial, with centered finite difference evaluation of the
%                    end-point time derivatives (i.e. a total of four temporal points are used).
%          - spatial_method  :  spatial interpolation and differentiation methods.
%                 - none      :  No spatial interpolation (value at the datapoint closest to each coordinate value).
%                 - lag4       :  4th-order Lagrange Polynomial interpolation along each spatial direction.
%                 - lag6       :  6th-order Lagrange Polynomial interpolation along each spatial direction.
%                 - lag8       :  8th-order Lagrange Polynomial interpolation along each spatial direction.
%                 - m1q4     :  Splines with smoothness 1 (3rd order) over 4 data points.
%                 - m2q8     :  Splines with smoothness 2 (5th order) over 8 data points.
%                 - m2q14    :  Splines with smoothness 2 (5th order) over 14 data points.
%                 - fd4noint  :  4th-order centered finite differencing (without spatial interpolation).
%                 - fd6noint  :  6th-order centered finite differencing (without spatial interpolation).
%                 - fd8noint  :  8th-order centered finite differencing (without spatial interpolation).
%                 - fd4lag4   :  4th-order Lagrange Polynomial interpolation in each direction, of the 4th-order finite difference values on the grid.
%           - spatial_operator  :  spatial interpolation and differentiation operator.
%                 - field         :  function evaluation & interpolation.
%                 - gradient   :  differentiation & interpolation.
%                 - hessian    :  differentiation & interpolation.
%                 - laplacian   :  differentiation & interpolation.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ----- Initialize getData parameters (except time and points) -----
variable = 'velocity';
temporal_method = 'none'; 
spatial_method = 'lag8';
spatial_operator  = 'field';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% example point distributions (2D plane, 3D box, random, time series) are provided below...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2D plane demo points : evenly spaced over a 2D plane lying along one of the primary axes
%     - time : the time to be queried (snapshot number for datasets without a full time evolution).
%     - nx, nz : number of points along each axis. total number of points queried will be n_points = nx * nz.
%     - x_points, y_points, z_points : point distributions along each axis, evenly spaced over the specified ranges.
%     - linspace(axis minimum, axis maximum, number of points).
%     - points : the points array evenly spaced out over the 2D plane.
%     - points array is instantiated as an empty array that will be filled inside the for loops.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

time = 1;

nx = 64;
nz = 64;
n_points = nx * nz;
 
points = zeros(n_points,3);

x_points = linspace(0.0, 0.4 * pi, nx);
y_points = 0.90;
z_points = linspace(0.0, 0.15 * pi, nz);

for i = 1 : nx
    for j = 1 :nz
       points(j +(i - 1) * nz, 1) = x_points(i);
       points(j +(i - 1) * nz, 2) = y_points;
       points(j +(i - 1) * nz, 3) = z_points(j);
    end
end

% ---- GetData ----
fprintf('\nRequesting %s at %i points...\n', variable, n_points);
result = getData(authkey, dataset, variable, time, temporal_method, spatial_method, spatial_operator, points);

if (nx >= 2) & (nz >= 2)
    % which component (column) of the data to plot (1-based index, so the first component is specified as 1).
    plot_component = 1;
    
    % ---- Display sample results on screen ----
    figure1 = figure('Color', [1 1 1], 'InvertHardcopy', 'off', 'PaperSize', [20.98 29.68]);
    axes1 = axes('FontSize', 16, 'LineWidth', 1.5, 'Parent', figure1, ...
        'XScale', 'lin', 'YScale', 'lin', 'Position', [0.18 0.18 0.76 0.76]); 
    box(axes1, 'on'); 
    hold(axes1, 'all'); 

    % Plotting data
    results = reshape(result(:,plot_component), [nz, nx]); 
    contourf(axes1, x_points, z_points, results, 300, 'LineColor','none');
    set(axes1, 'YDir', 'normal');
    colormap('hot')

    % Title and labels
    title([dataset, ' ', '(', variable, spatial_operator, ')'], 'FontSize', 15);
    xlabel(axes1, 'X');
    ylabel(axes1, 'Z');
    colorbar('FontSize', 16, 'Parent', figure1);
    set(axes1, 'DataAspectRatio', [1 1 1]); 
    axis tight; 
    set(axes1, 'XTickLabel', axes1.XTick, 'YTickLabel', axes1.YTick); 
    set(axes1, 'TickDir', 'out', 'TickLength', [0.02 0.02]);

    pause(1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3D box demo points : evenly spaced over a 3D volume
%     - time : the time to be queried (snapshot number for datasets without a full time evolution).
%     - nx,ny,nz : number of points along each axis. total number of points queried will be n_points= nx * ny * nz.
%     - x_points, y_points, z_points : point distributions along each axis, evenly spaced over the specified ranges.
%     - linspace(axis minimum, axis maximum, number of points).
%     - points : the points array evenly spaced out over the 3D volume.
%     - points array is instantiated as an empty array that will be filled inside the for loops.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

time = 1.0;

nx = 16;
ny = 16;
nz = 16;
n_points = nx * ny * nz;
 
points = zeros(n_points,3);

x_points = linspace(3.0, 3.3, nx);
y_points = linspace(-0.9, -0.6, ny);
z_points = linspace(0.2, 0.5, nz);

for i = 1:nx
    for j = 1:ny
        for k = 1:nz
            points((i - 1) * ny * nz + (j - 1) * nz + k, 1) = x_points(i);  
            points((i - 1) * ny * nz + (j - 1) * nz + k, 2) = y_points(j);
            points((i - 1) * ny * nz + (j - 1) * nz + k, 3) = z_points(k);
        end
    end
end

% ---- GetData ----
tic
fprintf('\nRequesting %s at %i points...\n', variable, n_points);
result = getData(authkey, dataset, variable, time, temporal_method, spatial_method, spatial_operator, points);
toc

if (nx >= 2) & (ny >= 2) & (nz >= 2)
    % Reshape result to 3D array
    results = reshape(result(:, 3), [nz, ny, nx]); 

    % ---- Display sample results on screen ----
    figure1 = figure('Color', [1 1 1], 'InvertHardcopy', 'off', 'PaperSize', [20.98 29.68]);

    axes1 = axes('FontSize', 16, 'LineWidth', 1.5, 'Parent', figure1, ...
        'XScale', 'lin', 'YScale', 'lin', 'ZScale', 'lin', ...
        'Position', [0.18 0.18 0.76 0.76]);
    box(axes1, 'on');
    hold(axes1, 'all'); 
    view(3); 

    % Generate 3D contour
    [X,Y,Z]=meshgrid(x_points, y_points, z_points);
    s =  slice(axes1, X, Y, Z, results, ...
          [x_points(1), x_points(end)], ...
          [y_points(1), y_points(end)], ...
          [z_points(1), z_points(end)]);

    shading interp;  

    set(axes1, 'YDir', 'normal');
    colormap('hot')
    colorbar('FontSize', 16, 'Parent', figure1);
    set(s,'EdgeColor','none')

    % Title and labels
    title([dataset, ' ', '(', variable, spatial_operator, ')'], 'FontSize', 15);
    xlabel(axes1, 'X');
    ylabel(axes1, 'Y');
    zlabel(axes1, 'Z');
    axis tight;
    set(axes1, 'XTickLabel', axes1.XTick, 'YTickLabel', axes1.YTick, 'ZTickLabel', axes1.ZTick);
    set(axes1, 'TickDir', 'out', 'TickLength', [0.02 0.02]);

    pause(1);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% random box demo points : uniformly distributed over the specified domain
%     - time : the time to be queried (snapshot number for datasets without a full time evolution).
%     - n_points : number of points.
%     - min_xyz, max_xyz : minimum and maximum (x, y, z) axes boundaries for generating the random distribution of points within.
%     - points : the points array containing a random distribution of points in the specified domain.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

time = 0.1;

n_points = 1000;

min_xyz  = [6.1359, -0.61359, 0.60];
max_xyz = [21.8656, 0.8656, 8.8656];

points = zeros(n_points, 3);
for i = 1:n_points
    points(i, 1) = rand() * (max_xyz(1) - min_xyz(1)) + min_xyz(1);
    points(i, 2) = rand() * (max_xyz(2) - min_xyz(2)) + min_xyz(2);
    points(i, 3) = rand() * (max_xyz(3) - min_xyz(3)) + min_xyz(3);
end

% ---- GetData ----
tic
fprintf('\nRequesting %s at %i points...\n', variable, n_points);
result = getData(authkey, dataset, variable, time, temporal_method, spatial_method, spatial_operator, points);
toc

% User-defined plot parameters.
% Which component (column) of the data to plot (1-based index, so the first component is specified as 1).
plot_component = 1;
% Number of bins for the histogram.
bins = 20;

x_plot = points(:, 1);
y_plot = points(:, 2);
z_plot = points(:, 3);
data_plot = result(:, plot_component);

% Plot the data.
fig = figure('Position', [100, 100, 1400, 500]);

% Scatter subplot.
ax_3d = subplot(1, 2, 1);
scatter3(ax_3d, x_plot, y_plot, z_plot, 20, data_plot, 'filled');
colormap(ax_3d, 'hot');
cbar = colorbar('Location', 'eastoutside');
cbar.Label.String = [variable, spatial_operator];
ax_3d.Title.String = 'Scatter';
ax_3d.XLabel.String = 'X';
ax_3d.YLabel.String = 'Y';
ax_3d.ZLabel.String = 'Z';
set(ax_3d, 'LineWidth', 1, 'FontSize', 16, 'TickLength', [0.02 0.02]);
rotate3d(ax_3d, 'on');

% Histogram subplot.
ax_hist = subplot(1, 2, 2);
histogram(ax_hist, data_plot, bins, 'FaceColor', 'green', 'EdgeColor', 'black');
ax_hist.Title.String = 'Histogram';
ax_hist.XLabel.String = [variable, spatial_operator];
ax_hist.YLabel.String = 'Count';
set(fig, 'Name', dataset, 'Color', 'w', 'Renderer', 'painters');
set(ax_hist, 'FontSize', 16, 'LineWidth', 1, 'TickLength', [0.02 0.02]);

pause(1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% time series demo point(s)
%     - time : the start time of the time series (snapshot number for datasets without a full time evolution).
%     - time_end : the end time of the time series (snapshot number for datasets without a full time evolution).
%     - delta_t : time step.
%     - points : the points array.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Temporal method.
temporal_method_tmp = 'pchip';
% Start time.
time_start = 0.1;
% End time.
time_end = 0.5;
% Time step.
delta_t = 0.008;

option = [time_end, delta_t];

points = [[10.33, 0.9, 4.6]];

result = getData(authkey, dataset, variable, time_start, temporal_method_tmp, spatial_method, spatial_operator, points, option);

% Show coordinates at time_end
if strcmp(variable, 'position')
    disp(result)
else
    % User-defined plot parameters.
    % Which component (column) of the data to plot (1-based index, so the first component is specified as 1).
    plot_component = 1;
    % which point of the data to plot (1-based index, so the first point component is specified as 1).
    point_component   = 1;
    times_plot = time_start : delta_t: time_end;

    % Plot the data.
    fig = figure('Position', [100, 100, 1000, 700]);
    signal = plot(times_plot, result(1: length(times_plot), point_component,  plot_component), 'Color', 'black', 'LineWidth', 3);
    title([dataset, ', time signal at (x,y,z) = ', '(', num2str(points(point_component,:)), ')' ]);
    xlabel('time');
    ylabel([variable, spatial_operator]);
    set(gca, 'FontSize', 16, 'LineWidth', 1,  'TickLength', [0.02 0.02]);
end