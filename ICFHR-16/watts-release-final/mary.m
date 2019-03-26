load matlab.mat

file_qid = fopen(['query_id' '.txt'],'w');
file_edit = fopen(['edit_dist' '.txt'],'w');
file_gt = fopen(['gt' '.txt'],'w');
file_pred = fopen(['pred' '.txt'],'w');

%for k=1:10
%    fwrite(file,lower(words{I(k)}));
%    fprintf(file,'\n');
%end
%fclose(file); 

for i=1:numel(cerfull)
    if(cerfull(i)==0)
        continue
    else
       fwrite(file_qid,num2str(qidx(i)));
       fprintf(file_qid,'\n');
       fwrite(file_edit,num2str(cerfull(i)));
       fprintf(file_edit,'\n');
       temp = char(lower(gtc{i}));
       fwrite(file_gt,temp);
       fprintf(file_gt,'\n');
       temp = char(lower(word{i}));
       fwrite(file_pred,temp);
       fprintf(file_pred,'\n');
    end
end
fclose(file_qid);
fclose(file_edit);
fclose(file_gt);
fclose(file_pred);