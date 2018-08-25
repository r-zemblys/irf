function plotResults(data,fix,filename,res)


Xdat = [];
Ydat = [];
klr  = {};
if isfield(data,'left')
    Xdat = [Xdat data.left.X(:)];
    Ydat = [Ydat data.left.Y(:)];
    klr = [klr {'g'}];
end
if isfield(data,'right')
    Xdat = [Xdat data.right.X(:)];
    Ydat = [Ydat data.right.Y(:)];
    klr = [klr {'r'}];
end
if isfield(data,'average') && ~isfield(data,'left') && ~isfield(data,'right')
    Xdat = [Xdat data.average.X(:)];
    Ydat = [Ydat data.average.Y(:)];
    klr = [klr {'b'}];
end
    

hf = figure(1);
set(hf,'Position',[res(1)/4 res(2)/4 res(1)/2 res(2)/2])

%% plot layout

myfontsize = 12;
fixlinewidth = 2;

h1 = subplot(2,1,1); hold on
for p=1:size(Xdat,2)
    plot(data.time,Xdat(:,p),[klr{p} '-']);
end
% add fixations
for b = 1:length(fix.startT)
    plot([fix.startT(b) fix.endT(b)],[fix.xpos(b) fix.xpos(b)],'k-','LineWidth',fixlinewidth);
end
hold off
ylabel('Horizontal position (pixels)','FontSize',myfontsize);
axis([0 data.time(end) 0 res(1)]);


h2 = subplot(2,1,2); hold on
for p=1:size(Ydat,2)
    plot(data.time,Ydat(:,p),[klr{p} '-']);
end
% add fixations
for b = 1:length(fix.startT)
    plot([fix.startT(b) fix.endT(b)],[fix.ypos(b) fix.ypos(b)],'k-','LineWidth',fixlinewidth);
end
hold off
ylabel('Vertical position (pixels)','FontSize',myfontsize);
xlabel('Time (ms)','FontSize',myfontsize);
axis([0 data.time(end) 0 res(2)]);

% link all x axes so zoom (change x limits) on one leads to same new x
% limits for all other ones
linkaxes([h1 h2],'x');

%% save and close
set(hf,'PaperPositionMode','auto');
drawnow;
% pause;
print(filename,'-dpng','-r300');
%close(hf);
