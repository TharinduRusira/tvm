#!/bin/bash
CURR_DIR=$(cd `dirname $0`; pwd)
<<<<<<< HEAD
APK_DIR=$CURR_DIR/../app/build/outputs/apk
=======
APK_DIR=$CURR_DIR/../app/build/outputs/apk/release
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
UNSIGNED_APK=$APK_DIR/app-release-unsigned.apk
SIGNED_APK=$APK_DIR/tvmdemo-release.apk
jarsigner -verbose -keystore $CURR_DIR/tvmdemo.keystore -signedjar $SIGNED_APK $UNSIGNED_APK 'tvmdemo'
echo $SIGNED_APK
