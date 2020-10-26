# This contains all the commands we ask them to run.

## declare all commands to test
declare -a arr=(
    "python main.py --help"
    "python main.py --obj ridge --opt closed-form"
    "python main.py --obj ridge --opt gd"
    "python main.py --obj lasso --opt gd"
    "python main.py --obj lasso --opt gd --init-lr 1e-1"
    "python main.py --obj lasso --opt gd --fix-lr"
    "python main.py --obj lasso --opt gd --fix-lr --init-lr 1e-1"
    "python main.py --obj smooth-lasso --opt gd"
    "python main.py --obj smooth-lasso --opt gd --temp 0.1"
    "python main.py --obj smooth-lasso --opt gd --init-lr 1e-1"
    "python main.py --obj smooth-lasso --opt gd --fix-lr"
    "python main.py --obj smooth-lasso --opt gd --fix-lr --init-lr 1e-1"
    "python main.py --dataset mini-mnist --obj logistic --opt gd"
    "python main.py --dataset mini-mnist --obj logistic --opt sgd"
    "python main.py --obj logistic --opt gd --epoch 1"
    "python main.py --obj logistic --opt sgd --epoch 1"
    "python main.py --dataset mini-mnist --opt sgd --obj svm"
    "python main.py --dataset mini-mnist --opt bcfw --obj svm"
)

## now loop through the above array
for i in "${arr[@]}"
do
   $i --no-visdom 2>/dev/null 1>/dev/null
   if [ $? -ne 0 ]; then
        echo "Command failed: $i"
        read -p "Press enter to re-run it and see the error"
        set -e
        $i --no-visdom
    fi
done
echo "Everything ran with no error"
