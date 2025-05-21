function result = getTurbineData(authToken, dataset, var, turbine_numbers, original_times)

    import matlab.net.http.*

    % convert turbine_numbers and times to JSON arrays
    turbines_json = jsonencode(turbine_numbers(:)');
    times_json = jsonencode(original_times(:)');

    % construct payload as a MATLAB struct
    json_payload = struct( ...
    'auth_token', authToken, ...
    'dataset_title', dataset, ...
    'turbine_variable', var, ...
    'turbines', turbine_numbers(:)', ...
    'turbine_times', original_times(:)');
    
    % send POST request
    headers = matlab.net.http.HeaderField('Content-Type', 'application/json');
    request = RequestMessage('POST', headers, json_payload);
    options = HTTPOptions('ConnectTimeout', 1000);

    url = 'https://web.idies.jhu.edu/turbulence-svc/turbine?include_metadata=0';
    response = request.send(url, options);

    % --- Print response and raw data for debugging ---
    %     disp('--- Raw HTTP Response ---');
    %     disp(response);
    
    % parse the JSON response
    raw_data = response.Body.Data;

    %     disp('--- Parsed JSON Data ---');
    %     disp(raw_data);
    %     disp('Data preview:');
    %     disp(raw_data(1));  % show first entry

    % --- error handling ---
    if response.StatusCode ~= matlab.net.http.StatusCode.OK
        if isfield(raw_data, 'description')
            error(['HTTP Error ', char(response.StatusCode), ': ', newline, ...
                   strjoin(raw_data.description, newline)]);
        else
            error(['HTTP Error ', char(response.StatusCode), '.']);
        end
    end
    
    % ----------------------------
    % convert to MATLAB table
    % convert raw numeric matrix to table
    result_table = array2table(raw_data, 'VariableNames', {'time', 'turbine', var});

    % convert turbine column to integer
    result_table.turbine = int32(result_table.turbine);

    % sort rows by turbine and time
    result_table = sortrows(result_table, {'turbine', 'time'});

    % add index column as first column
    num_times = numel(original_times);
    index_column = mod((0:height(result_table)-1)', num_times);

    % reorder columns: index, time, turbine, variable (otherwise index will be the last columns...)
    result = table(index_column, ...
               result_table.time, ...
               result_table.turbine, ...
               result_table.(var), ...
               'VariableNames', {'index', 'time', 'turbine', var});
end
