function merged = merge_structs(base, override)
% MERGE_STRUCTS  Merge two structs, with override taking precedence
%
%   merged = merge_structs(base, override) combines two structs, with fields
%   from <override> replacing any matching fields in <base>.
%
% -------------------------------------------------------------------------
% Inputs:
% -------------------------------------------------------------------------
%
% <base> - struct with default or base field values
%
% <override> - struct with fields that should replace those in <base>
%
% -------------------------------------------------------------------------
% Returns:
% -------------------------------------------------------------------------
%
% <merged> - struct containing all fields from <base>, with any fields
%   present in <override> replaced by their <override> values

    merged = base;
    fields = fieldnames(override);
    for i = 1:length(fields)
        merged.(fields{i}) = override.(fields{i});
    end
end
